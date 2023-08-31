#############################################
# @File    :   inference.py
# @Version :   1.0
# @Author  :   JiaweiDong
# @Time    :   2023/08/17  
# @Desc    :   KATS Project Inference Code
#############################################

import glob
import argparse
import numpy as np
import cv2
import torch
import json
from tqdm import tqdm
from os import path as osp
from timm_predictor import TimmPredictor

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser(description='Testing Config', add_help=False)
parser.add_argument('--model', default='faster_vit_0_224_crossattn', type=str)
parser.add_argument('--model-path', default='weights/TransientViT_0_v1.pth.tar', type=str)
parser.add_argument('--img-size', default=224, type=int)
parser.add_argument('--num-classes', default=2, type=int)
parser.add_argument('--device', default=0, type=str)
parser.add_argument('--img-src', default='images', type=str)
parser.add_argument('--out-path', default='output/result.json', type=str)
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()

args.device = int(args.device)

predictor = TimmPredictor(model_name=args.model, 
                          model_weights=args.model_path,
                          device=args.device, 
                          new_shape=args.img_size, 
                          num_classes=args.num_classes)
transform = predictor.transform

device = torch.device(args.device)

# 中心裁剪图像
def center_crop(img, new_size=48):
    h, w, c = img.shape
    new_w, new_h = new_size, new_size
    if w > new_w or h > new_h:
        left = (w - new_w)//2
        top = (h - new_h)//2
        img = img[top: top+new_h, left:left+new_w, :]
    return img

# 裁剪3x2或3x3大图
def crop_patch(img):
    h, w = img.shape
    row, col = 3, 3
    if h > w:
        col = 2
    grid_h, grid_w = h // row, w // col

    batch_imgs = []
    if col == 2:
        idxs_group = [[0,1]]
    elif col == 3:
        idxs_group = [[0,1], [0,2], [1,2]]
    for idxs in idxs_group:
        two_stack_img = []
        for i in idxs:
            stack_img = []
            for j in range(row):
                sub_img = img[j * grid_h:(j+1) * grid_h, i * grid_w:(i+1) * grid_w]
                stack_img.append(sub_img)

            stack_img = np.array(stack_img)
            stack_img = np.transpose(stack_img, (1,2,0))
            stack_img = stack_img[:, :, ::-1]
            # cv2.imwrite('tmp1.jpg', stack_img)
            stack_img = center_crop(stack_img)
            # cv2.imwrite('tmp2.jpg', stack_img)
            stack_img = transform(stack_img)
            two_stack_img.append(stack_img)

        # 6 224 224
        two_stack_img = torch.cat(two_stack_img, dim=0)
        batch_imgs.append(two_stack_img)

    # N 6 224 224
    batch_imgs = torch.stack(batch_imgs)
    return batch_imgs, idxs_group

def inference(img_src):
    results = []
    class_map = {True: 'real',
                 False: 'bogus'}
    
    if osp.isdir(img_src):
        img_list = glob.glob(img_src + '/*.jpg')
    elif osp.isfile(img_src):
        img_list = [img_src]
    else:
        raise NotImplementedError

    print(f'Testing at {img_src} Test Images Count: {len(img_list)}')

    for img_idx in tqdm(range(len(img_list))):
        img_path = img_list[img_idx]
        img_name = osp.basename(img_path)
        ori_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        batch_imgs, idx_group = crop_patch(ori_img)
        batch_imgs = batch_imgs.to(device)
        preds = predictor.inference(batch_imgs, preprocess=False)

        print('-' * 100)
        print(img_name)
        preds_voting = preds[:, 1] > 0.5
        preds_mean = np.mean(preds, axis=0)
        preds_mean_real = preds_mean[1]
        preds_mean_class = class_map[preds_mean_real > 0.5]

        preds_voting_class = np.sum(preds_voting) > (preds.shape[0] * 0.5)
        preds_voting_class = class_map[preds_voting_class]

        cross_inf_result = {}
        for idxs, pred, voting in zip(idx_group, preds, preds_voting):
            print(f'img cross attn idx: {idxs}, pred: {pred}')
            idxs = [str(x) for x in idxs]

            cur_cross_pred = pred.tolist()[1]
            cur_cross_pred = max(1-cur_cross_pred, cur_cross_pred)

            cross_inf_result[','.join(idxs)] = {'conf': cur_cross_pred,
                                                'vote': class_map[voting]}
            
        cur_class_conf = float(max(preds_mean_real, 1-preds_mean_real))

        print(f'ensemble mean result: {cur_class_conf}, \t class: {preds_mean_class}')
        print(f'ensemble voting: {preds_voting}, \t class: {preds_voting_class}')

        results.append({'file_name': img_name,
                        'class': preds_voting_class,
                        'conf': cur_class_conf,
                        'cross_inf_result': cross_inf_result})

        if args.test and img_idx == 20:
            break

    with open(args.out_path, 'w') as f:
        json.dump(results, f)

# 文件夹推理
inference(img_src=args.img_src)

# 单图片推理
# inference(img_src='rb_classify/1230719191550003741.jpg')