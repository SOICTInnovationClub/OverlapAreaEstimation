import argparse
import logging
import os
import cv2
from shapely import Polygon
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.utils import plot_img_and_mask


# Make Dataset
from utils.test_data_loading import OverlapDataset

path_to_predict = '/home/edtechai/Works/Pytorch-UNet/test_predict'

def predict_img(net,
                full_img,
                device,
                scale_factor=0.5,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(OverlapDataset.preprocess(None, full_img, scale_factor, is_mask=False, filename=''))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for a, v in enumerate(mask_values):
        out[mask == a] = v

    return Image.fromarray(out)

def find_maxpolygon(contours):
    max_pol, max_area = None, 0

    for contour in contours:
        hull1 = cv2.convexHull(contour)
        # cv2.fillConvexPoly(mask1, hull1, color=(255, 255, 255))

        hull1 = hull1.reshape(-1, 2)
        try:
            pol_area = Polygon(hull1).buffer(0).area
        except:
            pol_area = 0

        if max_area < pol_area:
            max_area = pol_area
            max_pol = hull1
            # breakpoint()

        # breakpoint()
    return max_pol


data_to_test = OverlapDataset("/home/edtechai/Works/Pytorch-UNet/data/test", 0.5)

net = UNet(n_channels=3, n_classes=2, bilinear=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading model /home/edtechai/Works/Pytorch-UNet/checkpoints/checkpoint_epoch10.pth')
print(f'Using device {device}')

net.to(device=device)
state_dict = torch.load('/home/edtechai/Works/Pytorch-UNet/checkpoints/checkpoint_epoch10.pth', map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)

logging.info('Model loaded!')

for a in range(len(data_to_test)):
    list_dir = data_to_test.list_dir[a]
    
    print("%.1f" % (1.4214214))