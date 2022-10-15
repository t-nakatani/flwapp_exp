import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import collections
from PIL import Image
import csv
import random
# import pandas as pd
import re
import random
import glob

class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, mode, batchsz, n_way, k_shot, k_query, resize, path_patch, startidx=0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        # self.patch_names = []
        # print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        # mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.cls_num = 2
        self.mode = mode
        # self.cross_val_idx = cross_val_idx
        self.create_batch(self.batchsz, mode, path_patch)

    def create_batch(self, batchsz, mode, path_patch):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        
        
        if mode != 'predict':
            print('===invalid mode===', mode)
        # with open(f'{path_patch}/flower_names.txt') as f:
        #     flower_names = [s.strip() for s in f.readlines()]
        flower_names = ['img']
        # if '.DS_Store' in flower_names:
        #     flower_names.remove('.DS_Store')
            # print(flower_names)
        for selected_flower in flower_names:  # for each batch
            # 1.select n_way classes randomly
            # selected_flower = selected_flower.replace('.png', '')
            support_x = [] # ex) [['0_10.png', '0_12.png'], ['0_0.png', '0_2.png']]
            query_x = []
            candi_s0 = glob.glob(f'{path_patch}/synthe/' + selected_flower + '*[02468].png')
            candi_s1 = glob.glob(f'{path_patch}/synthe/' + selected_flower + '*[13579].png')

                # 2. select k_shot + k_query for each class
            try:
                selected_imgs_spt0 = random.sample(candi_s0, self.k_shot)
                selected_imgs_spt1 = random.sample(candi_s1, self.k_shot)
            except:
                print(selected_flower + ' is not available')
                self.batchsz -= 1

                # print(mode)
                # print('--')
                print(candi_s0)
                print(candi_s1)
                # print('---')
                continue

            
            
            np.random.shuffle(selected_imgs_spt0) # いらん気がする
#             np.random.shuffle(selected_imgs_qry0)
            np.random.shuffle(selected_imgs_spt1)
#             np.random.shuffle(selected_imgs_qry1)
            for k in range(self.k_shot):
                support_x.append([selected_imgs_spt0[k], selected_imgs_spt1[k]])
                random.shuffle(support_x[-1])
            for k in range(1):
                query_x.append(sorted(glob.glob(f'{path_patch}/natural/' + selected_flower + '*.png')))
                # if selected_flower == '000_0':
                    # print(query_x)
                # random.shuffle(query_x[-1])
#             break

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
#             random.shuffle(query_x)
#             print(output)
#             print(query_x[0])

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets
        # print(support_x)
        # print(query_x)

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        # print('__getitem__')
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
#         if self.k_query > 1:
#             query_x = torch.FloatTensor(self.querysz//self.n_way, 3, self.resize, self.resize)
#         else:
#             query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
#         # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)
        
        path1 = ''
        path2 = ''
        flatten_support_x = [os.path.join(path1, item)
                             for sublist in self.support_x_batch[index] for item in sublist]

        support_y = np.array([int(re.findall('.+_(\d+).png', fname)[0])%2 for sublist in self.support_x_batch[index] for fname in sublist])

        flatten_query_x = [os.path.join(path2, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        # query_y = np.array([self.dic_qry[fname] for sublist in self.query_x_batch[index] for fname in sublist])
        query_x = torch.FloatTensor(len(flatten_query_x), 3, self.resize, self.resize)

        # print('relative:', support_y_relative, query_y_relative)
        # print(flatten_query_x)
        # self.patch_names = [re.findall('/(img_[0-9]+.png)', qry_x)[0] for qry_x in flatten_query_x] + ['']
#         print(flatten_query_x)
#         print(output)
        # with open('./maml_for_app/record/log_fname.txt', mode='a') as f:
        #     f.write(','.join(output))

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)
        # print(support_set_y)
        return support_x, torch.LongTensor(support_y), query_x#, patch_names

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=5, k_shot=1, k_query=1, batchsz=1000, resize=168)

    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_

        support_x = make_grid(support_x, nrow=2)
        query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())
        plt.pause(0.5)

        tb.add_image('support_x', support_x)
        tb.add_image('query_x', query_x)

        time.sleep(5)

    tb.close()
