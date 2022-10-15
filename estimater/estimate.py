import os, cv2, sys, re, math, Levenshtein, pickle, shutil, math, pprint, json
import numpy as np
import pandas as pd

sys.path.append('estimater/yolov5')
from estimater.yolov5 import detect
weight_path = 'estimater/yolov5/exp21_best.pt'

sys.path.append('estimater/maml')
from estimater.maml.load_model_predict import main
import argparse

def mk_data_dir():
    os.mkdir('./data')
    os.mkdir('./data/patch')
    os.mkdir('./data/patch/synthe')
    os.mkdir('./data/patch/for_synthe')
    os.mkdir('./data/patch/natural')
    return

def mk_patch_dir(dir):
    os.mkdir(f'{dir}/patch')
    os.mkdir(f'{dir}/patch/synthe')
    os.mkdir(f'{dir}/patch/for_synthe')
    os.mkdir(f'{dir}/patch/natural')
    return

def auto_grabcut(img, bb):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel,5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_ = img*mask2[:,:,np.newaxis]
    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 1, 255, 0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_p = 0
    idx = 0
    for i, cnt in enumerate(contours):
        if len(cnt > 10):
            p = cv2.arcLength(cnt,True)
            if max_p < p:
                max_p = p
                idx = i
    blk = np.zeros(img.shape, np.uint8)
    mask = cv2.drawContours(blk, [contours[idx]], -1, color=(1, 1, 1), thickness=-1)
    foreground = img * mask
    mask_ = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) * 255
    foreground = img * mask
    return foreground, mask_, [contours[idx]]



def Harris(gray_img):
    gray = np.float32(gray_img)
    dst = cv2.cornerHarris(gray,11,11,0.1)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.25*dst.max(),255,0)
    dst = np.uint8(dst)
    return(dst)

def find_IS_coordinate(center, radius, boundary_copy_img):
    '''
    is >> intersection
    
    input:
        center >> - of circle (x,y)
        radius >> - of circles
        boundary_copy_img >> -
    output:
        cx, cy >> coordinate of a circle center
        cx1, cy1 >> coordinate of intersection1
        cx2, cy2 >> coordinate of intersection2
    '''
    shape = boundary_copy_img.shape
    blank = np.zeros(shape, np.uint8)
    blank = cv2.circle(blank, center, radius, [255,255,0],3)
#     boundary_copy_img = cv2.circle(boundary_copy_img, center, radius, [255,255,0],1)


    # find IS
    intersection_img = cv2.bitwise_and(boundary_copy_img, blank)
    # gray_is = cv2.cvtColor(intersection_img, cv2.COLOR_BGR2GRAY)
    gray_is = intersection_img
    _, gray_is = cv2.threshold(gray_is, 1, 255, cv2.THRESH_BINARY)
    gray_is = cv2.dilate(gray_is, None)
    contours, hierarchy = cv2.findContours(gray_is, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    coord_list = []
    # intersections per one circle == 2
    if (len(contours)) < 2:
        return 0
    for k in range(2):
        M = cv2.moments(contours[k])
        if M['m00'] == 0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # contour重心(交点)をlistに
        coord_list.append((cx, cy))

    cx, cy = center
    cx1, cy1 = coord_list[0] 
    cx2, cy2 = coord_list[1]
    v1 = np.array([0, -1])
    v2 = np.array([(cx1 + cx2)/2 - cx, (cy1 + cy2)/2 - cy])

    if np.linalg.norm(v2) == 0:
        theta = 0
    else:
        cos_theta = np.inner(v1, v2) /  (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = math.acos(np.clip(cos_theta, -1.0, 1.0)) / math.pi * 180
        if cx < (cx1 + cx2)/2:
            theta *= -1 
    return [cx, cy, cx1, cy1, cx2, cy2, theta]

def convex(mask, c1, c2):
    x1 = int(c1[0]*0.8 + c2[0]*0.2)
    y1 = int(c1[1]*0.8 + c2[1]*0.2)    
    x2 = int(c1[0]*0.2 + c2[0]*0.8)    
    y2 = int(c1[1]*0.2 + c2[1]*0.8)
    x3 = int(c1[0]*0.5 + c2[0]*0.5)    
    y3 = int(c1[1]*0.5 + c2[1]*0.5)
    
    if mask[y1][x1] + mask[y2][x2] + mask[y3][x3] == 0:
        return False
    else: return True

def affine2(theta, img_, center):
    #回転角を指定
    angle = theta * (-1)
    #スケールを指定
    scale = 1.0
    #getRotationMatrix2D関数を使用
    center = list(map(int, center))
    trans = cv2.getRotationMatrix2D(center, angle , scale)
    #アフィン変換
    img_ = cv2.warpAffine(img_, trans, (img_.shape[1], img_.shape[0]))
    return img_

def extract_patches(img_copy, patch_name, C, theta, path_patch):
    l = 15
    path_natu = f'{path_patch}/natural/'
    cx, cy, cx1, cy1, cx2, cy2 = C
    img_ = affine2(theta, img_copy, (cx, cy))

    patch = cv2.resize(img_[cy-l:cy+l+30, cx-l-5:cx+l+5], None, fx = 0.5, fy = 0.3333333)
    cv2.imwrite(path_natu + patch_name, patch)
    cx1_ = int(cx + (cx1 - cx) * math.cos(math.radians(theta)) - (cy1 - cy) * math.sin(math.radians(theta)))
    cx2_ = int(cx + (cx2 - cx) * math.cos(math.radians(theta)) - (cy2 - cy) * math.sin(math.radians(theta)))
    cy1_ = int(cy + (cx1 - cx) * math.sin(math.radians(theta)) + (cy1 - cy) * math.cos(math.radians(theta)))
    cy2_ = int(cy + (cx2 - cx) * math.sin(math.radians(theta)) + (cy2 - cy) * math.cos(math.radians(theta)))
    patch1 = img_[cy1_-l:cy1_+l, cx1_-l:cx1_+l]
    patch2 = img_[cy2_-l:cy2_+l, cx2_-l:cx2_+l]
    if cx1_ > cx2_:
        patch2 = np.fliplr(patch2)
    else:
        patch1 = np.fliplr(patch1)
    
    return [patch1, patch2]

def create_patches_from_ptlist(ptlist, fname, path_patch, img_fore):
    """
    input:
        ptlist >> point coord list [(x1, y1), (x2, y2), ...]
        path_patch >> - path_patch
                                        - for_synthe
                                        - synthe
                                        - natural
    """
    path_natural = os.path.join(path_patch, 'natural')
    path_synthe = os.path.join(path_patch, 'synthe')
    path_for_synthe = os.path.join(path_patch, 'for_synthe')
    pnames = []
    for i, pt in enumerate(ptlist):
        img_gray = cv2.cvtColor(img_fore, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask,None)
        mask = cv2.dilate(mask,None)
        mask_ = mask + 0
        mask_ = cv2.erode(mask_,None)
        boundary = cv2.bitwise_xor(mask, mask_)

        for_synthe_patches = []
        img__ = img_fore + 0
        corners_list = []
        for i, pt in enumerate(ptlist):
            patch_name = fname.replace('.', '_' + str(i).zfill(2) + '.')
            pnames.append(patch_name)
            img_ = img_fore + 0
            boundary_ = boundary + 0
            radius = 18
            C = find_IS_coordinate(pt, radius, boundary_)
            if C == 0:
                continue
            cx, cy, cx1, cy1, cx2, cy2 , theta= C
            for_synthe_patches = for_synthe_patches + extract_patches(img_, patch_name, C[:-1], theta, path_patch)
            corners_list.append([fname, patch_name, cx, cy])
        create_synthe_pathches(for_synthe_patches, fname, path_patch)

    df_natural = pd.DataFrame(corners_list)
    df_natural.columns = ['fname', 'patch_name', 'cx', 'cy']
#     df.to_csv('data/natural_patches.csv', header=True, index=False)

    df_synthe = []
    for pname in pnames:
        fname = 'img.png'
        lr = int(re.findall('\d+', pname)[0]) % 2 # 0>>r, 1>>l
        df_synthe.append([pname, fname, lr])
    df_synthe = pd.DataFrame(df_synthe)
    df_synthe.columns = ['pname', 'fname', 'lr']
    # df_synthe.to_csv('../data/csv/synthe_patches.csv', header=True, index=False)
    return df_natural, df_synthe
    
def create_synthe_pathches(for_synthe_patches, fname, path_patch):
    path_synthe = f'{path_patch}/synthe/'
    for i, patch in enumerate(for_synthe_patches):
        cv2.imwrite(f'{path_patch}/for_synthe/' + fname.replace('.png', '') + '_' + str(i) + '.png', patch)
    cnt = 0
    for i, patch1 in enumerate(for_synthe_patches):
        for patch2 in for_synthe_patches[i+1:]:
            patch1_, patch2_ = synthesis(patch1, np.fliplr(patch2))
            pname = fname.replace('.', '_' + str(cnt).zfill(3) + '.')
            cv2.imwrite(path_synthe + pname, patch1_)
            cnt += 1
            pname = fname.replace('.', '_' + str(cnt).zfill(3) + '.')
            cv2.imwrite(path_synthe + pname, patch2_)
            cnt += 1
    return

def synthesis(img1, img2):
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _,gray_img1 = cv2.threshold(gray_img1,120,255,0)
    _,gray_img2 = cv2.threshold(gray_img2,120,255,0)
    not1 = cv2.bitwise_not(gray_img1)
    not2 = cv2.bitwise_not(gray_img2)
    
    bitwise_and1 = cv2.bitwise_and(img2, img2, mask = not1)
    bitwise_and2 = cv2.bitwise_and(img1, img1, mask = not2)
    img1_ = cv2.bitwise_and(img1, img1, mask = gray_img1)
    img2_ = cv2.bitwise_and(img2, img2, mask = gray_img2)

    synthetic_img_1on2 = cv2.bitwise_or(img1_, bitwise_and1)
    synthetic_img_2on1 = cv2.bitwise_or(img2_, bitwise_and2)
    
    return synthetic_img_1on2, synthetic_img_2on1

def extract_corners_and_create_patches(img, mask, path_img):
    dir, fname = os.path.split(path_img)
    path_patch = f'{dir}/patch'
    corners_list = []
    mask_ = mask + 0
    mask_ = cv2.erode(mask_,None)
    boundary = cv2.bitwise_xor(mask, mask_)
    harris = Harris(boundary)
    _, corners = cv2.threshold(harris, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(corners, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #cornerの重心座標を求める
    for_synthe_patches = []
    img__ = img + 0
    cnt = 0
    for i, contour in enumerate(contours):
        coord_list = []
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        patch_name = fname.replace('.', '_' + str(i).zfill(2) + '.')
        img_ = img + 0
        boundary_ = boundary + 0
        radius = 18
        C = find_IS_coordinate((cx, cy), radius, boundary_)
        if C == 0:
            continue
        cx, cy, cx1, cy1, cx2, cy2 , theta= C
        img__ = cv2.circle(img__, (cx,cy), 1, [0,255,0],5)
        img__ = cv2.circle(img__, (cx1,cy1), 1, [0,0,222],5)
        img__ = cv2.circle(img__, (cx2,cy2), 1, [0,0,222],5)

        if not convex(mask, (cx1, cy1), (cx2, cy2)):
            for_synthe_patches = for_synthe_patches + extract_patches(img_, patch_name, C[:-1], theta, path_patch)
            corners_list.append([fname, patch_name, cx, cy])
            cnt += 1
    create_synthe_pathches(for_synthe_patches, fname, path_patch)

    df_natural = pd.DataFrame(corners_list)
    df_natural.columns = ['fname', 'patch_name', 'cx', 'cy']
    pnames = sorted(os.listdir(f'{dir}/patch/synthe/'))
    if '.DS_Store' in pnames:
        pnames.remove('.DS_Store')

    df_synthe = []
    fname = 'img.png'
    for pname in pnames:
        lr = int(re.sub('.*_', '', pname).replace('.png', '')) % 2 # 0>>r, 1>>l
        df_synthe.append([pname, fname, lr])
    df_synthe = pd.DataFrame(df_synthe)
    df_synthe.columns = ['pname', 'fname', 'lr']
    return df_natural, df_synthe

def petal_array(dic_, df_):
    patchs = df_['patch_name']
    coords =  [(x, y) for x, y in zip(df_['cx'], df_['cy'])]
    cxcy=  (int(sum(df_['cx']) / len(df_['cx'])), int(sum(df_['cy']) / len(df_['cy'])))
    coords_ = [(xy[0] - cxcy[0], xy[1] - cxcy[1]) for xy in coords]
    list_ = sorted([[math.degrees(math.atan2(xy[1], xy[0])), pname] for xy, pname in zip(coords_, patchs)])
    preds = ''.join([dic_[_pnames[1]] for _pnames in list_])
    preds = preds.replace('0', 'r').replace('1', 'l')
    preds = preds[::-1] # countour clockwise >> clockwise

    return preds

def create_img_lr(img_lr, df):
    R = cv2.imread('media/saved_data/R.png')
    L = cv2.imread('media/saved_data/L.png')
    pnames = list(df['patch_name'])
    cx_list = list(df['cx'])
    cy_list = list(df['cy'])
    lr_list = list(df['label'])
    for cx, cy, lr in zip(cx_list, cy_list, lr_list):
        cv2.circle(img_lr, (cx, cy), 15, (255, 255, 255), thickness=-1)
        if int(lr) == 0:
            cv2.circle(img_lr, (cx, cy), 15, (0, 0, 255), thickness=2)
            img_lr[cy-10:cy+10, cx-10:cx+10] = R
        elif int(lr) == 1:
            cv2.circle(img_lr, (cx, cy), 15, (255, 0, 0), thickness=2)
            img_lr[cy-10:cy+10, cx-10:cx+10] = L
        else:
            print('type mismatching !!')
    return img_lr

def create_img_corner(img_corner, df):
    cx_list = list(df['cx'])
    cy_list = list(df['cy'])
    for cx, cy in zip(cx_list, cy_list):
        cv2.circle(img_corner, (cx, cy), 6, (255, 255, 0), thickness=-1)
    return img_corner


def predict_and_create_patch_dic(path_patch):
        argparser = argparse.ArgumentParser()
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
        argparser.add_argument('--weight_path', type=str, help='update steps for finetunning', default='estimater/maml/model')
        argparser.add_argument('--path_patch', type=str, help='path_to_patch_img', default=path_patch)

        args = argparser.parse_args(args=[])
        pnames, predicts = main(args)
        dic = dict([(pname, pred) for pname, pred in zip(pnames, predicts)])
        return dic

def rotate(str, i):
    str_ = str[i:] + str[:i]
    return str_

def LR2IEA(LR_arr):
    IEA_arr = []
    LR_arr_ = LR_arr + LR_arr[0]
    for i in range(len(LR_arr)):
        mini_arr = LR_arr_[i] + LR_arr_[i+1]
        if mini_arr == 'rl':
            IEA_arr.append('i')
        elif mini_arr == 'lr':
            IEA_arr.append('e')
        else:
            IEA_arr.append('a')
    return ''.join(IEA_arr)

def arr2TYPE(path_flw_dic, arr, cost):
    # cost : cost_dict of replace, delete, insert
    with open(path_flw_dic, 'rb') as f:
        dic = pickle.load(f)
    TYPE = list(dic.keys())
    min_ = 1000
    types = []
    for i, type_ in enumerate(TYPE):
        arr1 = dic[type_]
        for j in range(len(arr)):
            arr2 = rotate(arr, j)
            dist = 0
            for edit in Levenshtein.editops(arr1, arr2):
                dist += int(edit[0].replace('replace', str(cost['replace'])).
                            replace('delete', str(cost['delete'])).replace('insert', str(cost['insert'])))
            if dist < min_:
                types = [type_]
                min_ = dist
            elif dist == min_:
                types.append(type_)
    arr_ = arr[::-1]
    for i, type_ in enumerate(TYPE):
        arr1 = dic[type_]
        for j in range(len(arr_)):
            arr2 = rotate(arr_, j)
            dist = 0
            for edit in Levenshtein.editops(arr1, arr2):
                dist += int(edit[0].replace('replace', str(cost['replace'])).
                            replace('delete', str(cost['delete'])).replace('insert', str(cost['insert'])))
            if dist < min_:
                types = [type_]
                min_ = dist
            elif dist == min_:
                types.append(type_)
    return sorted(list(set(types))), min_


        
def calc_mean_point(ptlist):
    sum_x = 0
    sum_y = 0
    for pt in ptlist:
        sum_x += pt[0]
        sum_y += pt[1]
    mean_point = [sum_x//len(ptlist), sum_y//len(ptlist)]
    return mean_point

def internal_external_division_from_ptlist(ptlist):
    mean = calc_mean_point(ptlist)
    internal_division_ptlist = []
    external_division_ptlist = []
    for x, y in ptlist:
        x_ = int(mean[0] * 0.2 + x * 0.8)
        x__ = int(x * 1.3 - mean[0] * 0.3)
        y_ = int(mean[1] * 0.2 + y * 0.8)
        y__ = int(y * 1.3 - mean[1] * 0.3)
        dx = x - mean[0]
        dy = y - mean[1]
        tan_ = dy/dx
        atan = np.arctan(tan_)*180/math.pi
        if dx < 0:
            atan += 180
            if dy > 0:
                atan += 0
        elif dy < 0:
            atan += 360
            
        internal_division_ptlist.append([atan, x_, y_])
        external_division_ptlist.append([atan, x__, y__])
    return sorted(internal_division_ptlist), sorted(external_division_ptlist)

def calc_mean_points_for_seg (sorted_ptlist):
    new_ptlist = []
    sorted_ptlist.append(sorted_ptlist[0])
    for pt1, pt2 in zip(sorted_ptlist[:-1], sorted_ptlist[1:]):
        theta1, x1, y1 = pt1
        theta2, x2, y2 = pt2
        new_ptlist.append([int(x1/2+x2/2), int(y1/2+y2/2)])
        
    return new_ptlist

def make_mask_for_seg(foreground, ptlist):
    fore_ = foreground + 0
    _, gray_mask3c = cv2.threshold(foreground,1,1,cv2.THRESH_BINARY)
    cv2.imwrite('mask.png', gray_mask3c*255)
    ptlist_i, ptlist_e = internal_external_division_from_ptlist(ptlist)
    ptlist_i_ = calc_mean_points_for_seg (ptlist_i)
    for x, y in ptlist_i_:
        cv2.circle(gray_mask3c, (x, y), 5, (1, 0, 0), thickness=-1)
        cv2.circle(fore_, (x, y), 5, (255, 0, 0), thickness=-1)
    for theta, x, y in ptlist_e:
        cv2.circle(gray_mask3c, (x, y), 5, (0, 0, 0), thickness=-1)
        cv2.circle(fore_, (x, y), 5, (0, 255, 0), thickness=-1)
    gray_mask1c = gray_mask3c[:,:,0].astype('uint8')
    cv2.imwrite('fore_.png', fore_)
    return gray_mask1c

def mask2fore(old_fore, mask_gray):
#     mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blk = np.zeros(old_fore.shape, np.uint8)
    contours, hierarchy = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
    cv2.drawContours(blk, contours, -1, color=(1, 1, 1), thickness=-1)
    new_fore = old_fore * blk
    return new_fore

def re_grabcut(old_mask, old_fore):
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    new_mask, bgdModel, fgdModel = cv2.grabCut(old_fore,old_mask,None,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_MASK)
    new_fore = mask2fore(old_fore, new_mask*255)
    return new_fore

def infer_arr(path_img):
    mk_data_dir()
    dir, fname = os.path.split(path_img)
    path_patch = f'{dir}/patch'
    
    img = cv2.imread(path_img)
    cv2.imwrite('./data/img.png', img)
    bb_ = detect.run(weights=weight_path, source=path_img, nosave=True, imgsz=(256, 256)).tolist()[0]
    bb = [int(xy) for xy in bb_]
    img_bb = img + 0
    cv2.rectangle(img_bb, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 0), thickness=4)

    foreground, mask, contour4mask = auto_grabcut(img, bb)
    df_n, df_s = extract_corners_and_create_patches(img, mask, path_img) # df_natural, df_synthe

    dic = predict_and_create_patch_dic(path_patch)
    df_n['label'] = [dic[key] for key in df_n['patch_name']]

    arr_lr = petal_array(dic, df_n)

    img_corner = np.copy(img)
    img_corner = create_img_corner(img_corner, df_n)
    img_lr = np.copy(img)
    img_lr = create_img_lr(img_lr, df_n)
    cost = {'replace':1, 'delete':1, 'insert':1}
    path_flw_dic = 'media/saved_data/dic_iea.pkl'
    arr_iea = LR2IEA(arr_lr)
    types, min_ = arr2TYPE(path_flw_dic, arr_iea, cost)
    ARR = arr_iea.upper()

    cv2.imwrite('./data/img_bb.png', img_bb)
    cv2.imwrite('./data/img_fore.png', foreground)
    cv2.imwrite('./data/img_corner_.png', img_corner)
    cv2.imwrite('./data/img_lr.png', img_lr)
    df_n.to_csv('./data/df_n.csv', index=False)
    # print(type(bb), type(contour4mask), type(contour4mask[0]), type(contour4mask[0][0]))
    log_result = {'fname': os.path.basename(path_img), 'bb': bb, 'contour': list(map(int, np.array(contour4mask).flatten())), 'arrangement': ARR}

    with open("./data/log_result.json", "w") as f:
        json.dump(log_result, f, indent=4)
    return bb, contour4mask, df_n, ARR

def min_dist_arg(coord, coord_list):
    dist_list = [(coord[0]-coord2[0])**2 + (coord[1]-coord2[1])**2 for coord2 in coord_list]
    return np.argmin(dist_list)

def update_intersection_label(path_img, clicked_coord):
    dir, fname = os.path.split(path_img)
    df_n = pd.read_csv(f'{dir}/df_n.csv')
    coord_list = np.array(df_n.loc[:, 'cx':'cy'])
    clicked_coord = np.array(clicked_coord).astype(np.float64)

    for coord in clicked_coord:
        # coord *= SIZE_CHANGING_RATIO
        idx = min_dist_arg(coord, coord_list)
        df_n.loc[idx, 'label'] = 1 - df_n.loc[idx, 'label'] # flip label
    img = cv2.imread(path_img)
    img_ = create_img_lr(img, df_n)
    shutil.move(f'{dir}/img_lr.png', f'{dir}/img_lr_old.png')
    cv2.imwrite(f'{dir}/img_lr.png', img)
    df_n.to_csv(f'{dir}/df_n.csv', index=False)
    return

# 一回目の座標とのマッチングを行うか，修正の際に指定された座標を信じるか． 簡単な後者をとりあえず実装．
def re_infer_with_clicked(path_img, clicked_coord_xy):
    # create df_corner, then update df_n(df for natural_patch) by df_corner
    dir, fname = os.path.split(path_img)
    path_patch = f'{dir}/patch'
    if os.path.exists(f'{dir}/patch'):
        shutil.rmtree(f'{dir}/patch')
    mk_patch_dir(dir)
    img = cv2.imread(path_img)
    img_fore = cv2.imread(f'{dir}/img_fore.png')
    df_n, df_s = create_patches_from_ptlist(clicked_coord_xy, fname, f'{dir}/patch', img_fore)
    dic = predict_and_create_patch_dic(path_patch)
    df_n['label'] = [dic[key] for key in df_n['patch_name']]

    arr_lr = petal_array(dic, df_n)
    img_corner = np.copy(img)
    img_lr = np.copy(img)
    img_corner = create_img_corner(img_corner, df_n)
    img_lr = create_img_lr(img_lr, df_n)
    cost = {'replace':1, 'delete':1, 'insert':1}
    path_flw_dic = 'media/saved_data/dic_iea.pkl'
    arr_iea = LR2IEA(arr_lr)
    types, min_ = arr2TYPE(path_flw_dic, arr_iea, cost)
    ARR = arr_iea.upper()

    shutil.move(f'{dir}/img_corner_.png', f'{dir}/img_corner_old.png')
    cv2.imwrite(f'{dir}/img_corner_.png', img_corner)
    cv2.imwrite(f'{dir}/img_lr.png', img_lr)
    df_n.to_csv(f'{dir}/df_n.csv', index=False)

    return
