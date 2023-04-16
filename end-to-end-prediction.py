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
    
    frame = int(os.path.basename(list_dir[0])[5:-4])
    scenc_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(list_dir[0]))))
    
    cam1_state = False
    cam2_state = False
    cam3_state = False
    cam4_state = False
    
    if list_dir[0] != 'None':
        cam1_state = True
    if list_dir[1] != 'None':
        cam2_state = True
    if list_dir[2] != 'None':
        cam3_state = True
    if list_dir[3] != 'None':
        cam4_state = True

    img = data_to_test[a]

    
    start_time = time()
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=0.5,
                       out_threshold=0.5,
                       device=device)
    
    # Đọc ảnh binary
    out_filename = 'final.jpg'
    result = mask_to_image(mask, mask_values)
    result.save(out_filename)
    image = cv2.imread('final.jpg', cv2.IMREAD_GRAYSCALE)
    # lấy kích thước ảnh
    height, width = image.shape[:2]

    # cắt ảnh thành 4 phần bằng cách chia đôi chiều rộng và chiều cao
    x_center = width // 2
    y_center = height // 2

    cam1 = image[0:y_center, 0:x_center]
    cam2 = image[0:y_center, x_center:width]
    cam3 = image[y_center:height, 0:x_center]
    cam4 = image[y_center:height, x_center:width]
    # Tìm contours
    contours1, hierarchy1 = cv2.findContours(cam1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(cam2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours3, hierarchy3 = cv2.findContours(cam3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours4, hierarchy4 = cv2.findContours(cam4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask1 = np.zeros_like(cam1)
    mask2 = np.zeros_like(cam2)
    mask3 = np.zeros_like(cam3)
    mask4 = np.zeros_like(cam4)

    pol1 = find_maxpolygon(contours1)
    pol2 = find_maxpolygon(contours2)
    pol3 = find_maxpolygon(contours3)
    pol4 = find_maxpolygon(contours4)
    
    predict_time = time() - start_time
    FPS = float(1/predict_time)
    
    try:
        os.mkdir(os.path.join(path_to_predict, scenc_name))
    except:
        pass
    
    if cam1_state:
        f = open(os.path.join(path_to_predict, scenc_name, "CAM_1.txt"), "a")
        cam1_s = "frame_" + str(frame) + ".jpg, ("
        for i in range(len(pol1)):
            cam1_s += str(pol1[i][0]) + ',' + str(pol1[i][1]) + ','
        cam1_s = cam1_s[:-1] + "), "
        cam1_s += "%.1f" % (FPS)
        f.write(f"{cam1_s}\n")
        f.close()
        
    if cam2_state:
        f = open(os.path.join(path_to_predict, scenc_name, "CAM_2.txt"), "a")
        cam2_s = "frame_" + str(frame) + ".jpg, ("
        for i in range(len(pol2)):
            cam2_s += str(pol2[i][0]) + ',' + str(pol2[i][1]) + ','
        cam2_s = cam2_s[:-1] + "), "
        cam2_s += "%.1f" % (FPS)
        f.write(f"{cam2_s}\n")
        f.close()
        
    if cam3_state:
        f = open(os.path.join(path_to_predict, scenc_name, "CAM_3.txt"), "a")
        cam3_s = "frame_" + str(frame) + ".jpg, ("
        for i in range(len(pol3)):
            cam3_s += str(pol3[i][0]) + ',' + str(pol3[i][1]) + ','
        cam3_s = cam3_s[:-1] + "), "
        cam3_s += "%.1f" % (FPS)
        f.write(f"{cam3_s}\n")
        f.close()
        
    if cam4_state:
        f = open(os.path.join(path_to_predict, scenc_name, "CAM_4.txt"), "a")
        cam4_s = "frame_" + str(frame) + ".jpg, ("
        for i in range(len(pol4)):
            cam4_s += str(pol4[i][0]) + ',' + str(pol4[i][1]) + ','
        cam4_s = cam4_s[:-1] + "), "
        cam4_s += "%.1f" % (FPS)
        f.write(f"{cam4_s}\n")
        f.close()
    

    # logging.info(f'Mask saved to {out_filename}')
        
        

    



