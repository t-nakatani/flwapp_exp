import  numpy as np
import  torch, os, re
from MiniImagenet_for_prediction import MiniImagenet
# from    MiniImagenet2 import MiniImagenet2
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
import pandas as pd

from meta_prediction import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]
    # cuda = 'cuda:1'
    # cuda = 'cuda:' + args.gpu_index
    device = torch.device('cpu')
    maml = Meta(args, config).to(device)

    maml = torch.load(args.weight_path, map_location=torch.device('cpu'))
    # print('=== finished loading model===')
    # print(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
#     print(maml)
    # print('Total trainable tensors:', num)
    # if args.mode == 0:
    #     mode_val_test = 'val'
    #     train = 'train'
    # else:
    #     mode_val_test = 'test'
    #     train = 'train_ts'

    # batchsz here means total episode number
    mini_test = MiniImagenet(mode='predict', n_way=2, k_shot=args.k_spt, k_query=args.k_qry, batchsz=1, resize=args.imgsz, path_patch=args.path_patch)

    db_test = DataLoader(mini_test, 1, shuffle=False, num_workers=0, pin_memory=True)
    # print('=== finished loading data===')
    patch_names = [re.findall('/(img_[0-9]+.png)', qry_x)[0] for qry_x in mini_test.query_x_batch[0][0]]
    # print(patch_names)
    # print(db_test)
    # predicts_ = []
    for x_spt, y_spt, x_qry, in db_test:
        # print(x_spt.shape)
        x_spt, y_spt, x_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
         x_qry.squeeze(0).to(device)

        predicts_ = maml.predict(x_spt, y_spt, x_qry)
        # pred.append('')
        # with open('./maml_for_app/record/log_pred.txt', mode='a') as f:
        #     f.write(','.join(pred))
    # print(mini_test.query_x_batch[0][0])
    patch_names = [re.findall('/(img_[0-9]+.png)', qry_x)[0] for qry_x in mini_test.query_x_batch[0][0]]
    # print(predicts_)
    # predicts = [int(label) for label in predicts_]
    return patch_names, predicts_
                    

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--epoch', type=int, help='epoch number', default=80000)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=20)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
    # argparser.add_argument('--gpu_index', type=str, help='index of gpu', default='0')
    # argparser.add_argument('--mode', type=int, help='mode: 0 for val, 1 for test', default=0)
    # argparser.add_argument('--cross_val_idx', type=int, help='set this to val idx', default=0)
    # argparser.add_argument('--record_dir', type=str, help='', default='0')
    # argparser.add_argument('--patch_num', type=str, help='', default='1250')
    

    args = argparser.parse_args()

    main(args)
