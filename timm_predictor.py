import timm
#############################################
# @File    :   yolov7_predictor.py
# @Version :   1.0
# @Author  :   JiaweiDong
# @Time    :   2022/10/10 Mon  
# @Desc    :   
#############################################
import time
import os
import cv2
from copy import deepcopy
import sys
sys.path.append("prime_models")
from collections import OrderedDict, defaultdict
import torch
import timm
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from loguru import logger
import torch.nn.functional as F

class TimmPredictor:
    def __init__(self, model_name,
                 model_weights, 
                 device=None, 
                 new_shape=224, 
                 num_classes=2):
        self.new_shape = new_shape

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"[TIMM CLS Model] --------- Load from: {model_weights} ---------")
        logger.info(f"[TIMM CLS Model] --------- DEVICES: {device} ---------")
        logger.info(f"[TIMM CLS Model] --------- SHAPE: {self.new_shape} ---------")
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)


        state = torch.load(model_weights, map_location='cpu')
        if 'state_dict_ema' in state:
            logger.debug('EMA Model Found, Using EMA...')
            state = state['state_dict_ema']
        elif 'state_dict' in state:
            logger.debug('EMA Model Not Found, Using Normal...')
            state = state['state_dict']
        else:
            logger.error('No State Dict Found...')
        state = self.clean_state_dict(state)
        self.model.load_state_dict(state, strict=False)
        # self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("[TIMM CLS Model] --------- Load Success ---------")

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([self.new_shape,self.new_shape]),
            # transforms.CenterCrop(320), 
            transforms.ToTensor(),            
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def clean_state_dict(self, state_dict):
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = k[12:] if k.startswith('module.base.') else k
            k = k[6:] if k.startswith('model.') else k
            # k = k.replace('fc.', 'head.') if k.startswith('fc.') else k
            # k = k.replace('aux_bn.', '')
            cleaned_state_dict[k] = v
        return cleaned_state_dict

    def preprocess_single(self, img):
        img = img[:, :, ::-1]
        img = self.transform(img)
        img = torch.unsqueeze(img, 0)
        return img.to(self.device)

    def preprocess_multi(self, imgs):
        transformed_imgs = []
        for img in imgs:
            img = img[:, :, ::-1]
            ti = self.transform(img)
            # ti = ti.type(torch.cuda.FloatTensor)
            transformed_imgs.append(ti)
        transformed_imgs = torch.stack(transformed_imgs)
        return transformed_imgs.to(self.device)

    def inference(self, imgs, preprocess=True):
        if preprocess:
            if isinstance(imgs, list):
                imgs = self.preprocess_multi(imgs)
            elif isinstance(imgs, np.ndarray):
                imgs = self.preprocess_single(imgs)
            else:
                raise NotImplementedError('This dimension is not implemented.')
        with torch.no_grad():
            preds = self.model(imgs)
            # print(preds)
            if len(preds.size()) == 1:
                preds = preds.unsqueeze(dim=0)
            preds = torch.softmax(preds, dim=1)
        preds = self.postprocess(preds)
        return preds

    def postprocess(self, preds):   
        preds = preds.detach().cpu().numpy()
        return preds

