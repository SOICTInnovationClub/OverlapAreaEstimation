import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon
import cv2
import torch
from PIL import Image, ImageOps
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

def load_mask(filename):
    mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # apply thresholding to convert grayscale to binary image
    ret, mask = cv2.threshold(mask, 125, 1, 0)
    return mask

class OverlapDataset(Dataset):
    def __init__(self, dataset_dir, scale: float = 1.0) -> None:
        super().__init__()
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        
        self.list_dir = []
        for scene in os.listdir(dataset_dir):
            cam1 = False
            cam2 = False
            cam3 = False
            cam4 = False
            
            
            if os.path.exists(os.path.join(dataset_dir, scene, 'CAM1')):
                cam1 = True
            if os.path.exists(os.path.join(dataset_dir, scene, 'CAM2')):
                cam2 = True
            if os.path.exists(os.path.join(dataset_dir, scene, 'CAM3')):
                cam3 = True
            if os.path.exists(os.path.join(dataset_dir, scene, 'CAM4')):
                cam4 = True

            number_of_frame = len(os.listdir(os.path.join(dataset_dir, scene, 'CAM1')))
            
            for i in range(number_of_frame):
                list_frame = []
                if cam1:
                    list_frame.append(os.path.join(dataset_dir, scene, 'CAM1', f'frame{i}.png'))
                    # list_frame.append(os.path.join(dataset_dir, scene, 'CAM1', 'polygon', f'img{i}.png'))
                else:
                    list_frame.append('None')
                    # list_frame.append('None')
                    
                if cam2:
                    list_frame.append(os.path.join(dataset_dir, scene, 'CAM2' , f'frame{i}.png'))
                    # list_frame.append(os.path.join(dataset_dir, scene, 'CAM2', 'polygon', f'img{i}.png'))
                else:
                    list_frame.append('None')
                    # list_frame.append('None')
                    
                if cam3:
                    list_frame.append(os.path.join(dataset_dir, scene, 'CAM3' , f'frame{i}.png'))
                    # list_frame.append(os.path.join(dataset_dir, scene, 'CAM3', 'polygon', f'img{i}.png'))
                else:
                    list_frame.append('None')
                    # list_frame.append('None')
                    
                if cam4:
                    list_frame.append(os.path.join(dataset_dir, scene, 'CAM4', f'frame{i}.png'))
                    # list_frame.append(os.path.join(dataset_dir, scene, 'CAM4', 'polygon', f'img{i}.png'))
                else:
                    list_frame.append('None')
                    # list_frame.append('None')

                self.list_dir.append(list_frame)

        self.mask_values = [0, 255]
        
    def __len__(self):
        return len(self.list_dir)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask, filename):
        if is_mask:
            pil_img = ImageOps.invert(pil_img)
        
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # pil_img.save(filename)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img
        
    def __getitem__(self, idx):
        frame = self.list_dir[idx]
        
        img_list = []
        # mask_list = []
        for i in range(0, 4, 1):
            if frame[0 + i] != "None":
                img_list.append(load_image(frame[0 + i]))
                # mask_list.append(load_image(frame[1 + i]))
                
            else:
                img_list.append(load_image("/home/edtechai/Works/Pytorch-UNet/data/black.png"))
                # mask_list.append(load_image("/home/edtechai/Works/Pytorch-UNet/data/white.png"))  
        
        img = get_concat_v(get_concat_h(img_list[0], img_list[1]), get_concat_h(img_list[2], img_list[3]))
        # mask = get_concat_v(get_concat_h(mask_list[0], mask_list[1]), get_concat_h(mask_list[2], mask_list[3]))
        
        # img = self.preprocess(self.mask_values, img, self.scale, is_mask=False, filename='img.jpg')
        # mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True, filename='ground.jpg')
        
        # assert img.size == mask.size, \
        #         f'Image and mask should be the same size, but are {img.size} and {mask.size}'
        
        return img
        
        # return {
        #     'image': torch.as_tensor(img.copy()).float().contiguous(),
        #     'mask': torch.as_tensor(mask.copy()).long().contiguous()
        # }