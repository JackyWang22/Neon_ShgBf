import os
import json
import torch
import random
import pandas as pd
import numpy as np
import nibabel as nib
import torchvision.transforms.functional as TF
import nibabel as nib

import cv2
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import Dataset
from loaders.bezier_aug import stochastic_intensity_transformation

def get_nonwhite_msk(pil_image, delta=0):

    image_np = np.array(pil_image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    delta = min(delta, 132)
    lower_grey_white = np.array([0, 0, 132-delta])  # Lower bound of saturation and a reasonably high value
    upper_grey_white = np.array([180, 50, 255])  # Upper bound of saturation is low, high value

    grey_white_mask = cv2.inRange(hsv_image, lower_grey_white, upper_grey_white)
    grey_white_mask = ~grey_white_mask

    if grey_white_mask.sum() < 5:
        get_nonwhite_msk(pil_image, delta=delta+5)

    return grey_white_mask

class ImageTransform:

    def __init__(self):
        pass

    def __call__(self, mov_img, fix_img, mov_msk, fix_msk, is_training=True):

        mov_img = TF.to_tensor(mov_img)
        fix_img = TF.to_tensor(fix_img)

        mov_msk = torch.tensor(mov_msk).float().unsqueeze(0)
        fix_msk = torch.tensor(fix_msk).float().unsqueeze(0)

        mov_max = F.adaptive_max_pool2d(mov_img, (1,1))
        mov_min = -F.adaptive_max_pool2d(-mov_img, (1,1))
        mov_img = (mov_img - mov_min) / (mov_max - mov_min)

        fix_max = F.adaptive_max_pool2d(fix_img, (1,1))
        fix_min = -F.adaptive_max_pool2d(-fix_img, (1,1))
        fix_img = (fix_img - fix_min) / (fix_max - fix_min)

        if is_training and random.random() > 0.5:
            # random flip
            if random.random() > 0.5:
                mov_img = torch.flip(mov_img, [1])
                fix_img = torch.flip(fix_img, [1])
                mov_msk = torch.flip(mov_msk, [1])
                fix_msk = torch.flip(fix_msk, [1])
            if random.random() > 0.5:
                mov_img = torch.flip(mov_img, [2])
                fix_img = torch.flip(fix_img, [2])
                mov_msk = torch.flip(mov_msk, [2])
                fix_msk = torch.flip(fix_msk, [2])

            mov_img = stochastic_intensity_transformation(mov_img.unsqueeze(0)).squeeze(0)
            fix_img = stochastic_intensity_transformation(fix_img.unsqueeze(0)).squeeze(0)

            mov_max = F.adaptive_max_pool2d(mov_img, (1,1))
            mov_min = -F.adaptive_max_pool2d(-mov_img, (1,1))
            mov_img = (mov_img - mov_min) / (mov_max - mov_min)

            fix_max = F.adaptive_max_pool2d(fix_img, (1,1))
            fix_min = -F.adaptive_max_pool2d(-fix_img, (1,1))
            fix_img = (fix_img - fix_min) / (fix_max - fix_min)

        return mov_img, fix_img, mov_msk, fix_msk

class ShgBfReg_loader(Dataset): # ShgBfReg_loader

    affines = {}
    def __init__(self,
            root_dir = '/home/jackywang/Documents/Datasets/COMULISSHGBF/imagesTr',
            split = 'train', # train, val or test
        ):
        self.root_dir = root_dir
        self.split = split

        json_dir = os.path.join(root_dir, 'COMULISSHGBF_dataset.json')
        json_data = json.load(open(json_dir, 'r'))

        if split == 'train':
            self.root_dir = os.path.join(root_dir, 'train')
            data_list = os.listdir(self.root_dir)
            data_list = [f for f in data_list if f.endswith('SHG.tif')]
            self.init_training_pairs(data_list)
        elif split == 'val':
            self.root_dir = os.path.join(root_dir, 'val')
            data_list = os.listdir(self.root_dir)
            data_list = [f for f in data_list if f.endswith('.csv')]
            self.init_validation_pairs(data_list)
        elif split == 'test':
            self.root_dir = root_dir
            data_list = json_data['registration_test']
            self.init_testing_pairs(data_list)
        else:
            raise NotImplementedError

        self.transform = ImageTransform()

    def init_testing_pairs(self, data_list):

        self.data_list = []
        for idx in range(len(data_list)):
            mov_fp = os.path.join(self.root_dir, data_list[idx]['moving'])
            fix_fp = os.path.join(self.root_dir, data_list[idx]['fixed'])
            mov_img = nib.load(mov_fp).get_fdata()
            fix_img = nib.load(fix_fp).get_fdata()

            mov_img = Image.fromarray(np.squeeze((mov_img)).astype(np.uint8))
            fix_img = Image.fromarray(np.squeeze((fix_img)).astype(np.uint8))
            mov_msk = get_nonwhite_msk(mov_img)
            fix_msk = np.array(fix_img) >= 1
            self.data_list.append({
                'moving': mov_img, 
                'fixed': fix_img, 
                'moving_msk': mov_msk, 
                'fixed_msk': fix_msk,
                'idx': int(mov_fp.split('COMULISSHGBF_')[-1].split('_0001')[0]),
            })
            print('Loaded in memory %d/%d' % (idx+1, len(data_list)), end='\r', flush=True)
        print("Total number of images loaded: %d" % len(self.data_list))

    def init_training_pairs(self, data_list):
        self.pair_list = []
        for idx in range(len(data_list)):
            fixed = data_list[idx]
            moving = data_list[idx].replace('SHG.tif', 'BF.tif')
            self.pair_list.append({
                'moving': moving,
                'fixed': fixed,
            })
        self.data_list = []
        for idx in range(len(data_list)):
            mov_fp = os.path.join(self.root_dir, self.pair_list[idx]['moving'])
            fix_fp = os.path.join(self.root_dir, self.pair_list[idx]['fixed'])
            mov_img = Image.open(mov_fp)
            fix_img = Image.open(fix_fp)
            mov_msk = get_nonwhite_msk(mov_img)
            fix_msk = np.array(fix_img) >= 1
            self.data_list.append({'moving': mov_img, 'fixed': fix_img, 'moving_msk': mov_msk, 'fixed_msk': fix_msk})
            print('Loaded in memory %d/%d' % (idx+1, len(data_list)), end='\r', flush=True)
        print("Total number of images loaded: %d" % len(self.data_list))

    def init_validation_pairs(self, data_list):
        self.pair_list = []
        for idx in range(len(data_list)):
            fixed = data_list[idx].replace('.csv', '_SHG.tif')
            moving = data_list[idx].replace('.csv', '_transformed.tif')
            coords = data_list[idx]
            self.pair_list.append({
                'moving': moving,
                'fixed': fixed,
                'coords': coords,
            })
        self.data_list = []
        for idx in range(len(data_list)):
            mov_fp = os.path.join(self.root_dir, self.pair_list[idx]['moving'])
            fix_fp = os.path.join(self.root_dir, self.pair_list[idx]['fixed'])
            coords = pd.read_csv(os.path.join(self.root_dir, self.pair_list[idx]['coords']))

            mov_x = torch.tensor(coords['mov_X'].values).float()
            mov_y = torch.tensor(coords['mov_y'].values).float()
            fix_x = torch.tensor(coords['fix_x'].values).float()
            fix_y = torch.tensor(coords['fix_y'].values).float()

            mov_coords = torch.stack([mov_x, mov_y], dim=1)
            fix_coords = torch.stack([fix_x, fix_y], dim=1)

            mov_img = Image.open(mov_fp)
            fix_img = Image.open(fix_fp)

            mov_msk = get_nonwhite_msk(mov_img)
            fix_msk = np.array(fix_img) >= 1

            self.data_list.append({
                'moving': mov_img,
                'fixed': fix_img,
                'moving_coords': mov_coords,
                'fixed_coords': fix_coords,
                'moving_msk': mov_msk,
                'fixed_msk': fix_msk,
            })
            print('Loaded in memory %d/%d' % (idx+1, len(data_list)), end='\r', flush=True)
        print("Total number of images loaded: %d" % len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def get_train_item(self, idx):
        mov_img = self.data_list[idx]['moving']
        fix_img = self.data_list[idx]['fixed']
        mov_msk = self.data_list[idx]['moving_msk']
        fix_msk = self.data_list[idx]['fixed_msk']

        mov_img, fix_img, mov_msk, fix_msk = self.transform(mov_img, fix_img, mov_msk, fix_msk, is_training=True)

        return mov_img, fix_img, mov_msk, fix_msk

    def get_test_item(self, idx):
        mov_img = self.data_list[idx]['moving']
        fix_img = self.data_list[idx]['fixed']
        mov_msk = self.data_list[idx]['moving_msk']
        fix_msk = self.data_list[idx]['fixed_msk']
        idx = self.data_list[idx]['idx']

        mov_img, fix_img, mov_msk, fix_msk = self.transform(mov_img, fix_img, mov_msk, fix_msk, is_training=False)

        return mov_img, fix_img, mov_msk, fix_msk, idx

    def get_val_item(self, idx):

        mov_img = self.data_list[idx]['moving']
        fix_img = self.data_list[idx]['fixed']
        mov_coords = self.data_list[idx]['moving_coords']
        fix_coords = self.data_list[idx]['fixed_coords']
        mov_msk = self.data_list[idx]['moving_msk']
        fix_msk = self.data_list[idx]['fixed_msk']

        mov_img, fix_img, mov_msk, fix_msk = self.transform(mov_img, fix_img, mov_msk, fix_msk, is_training=False)

        return mov_img, fix_img, mov_coords, fix_coords, mov_msk, fix_msk, idx

    def __getitem__(self, idx):

        if self.split == 'val':
            return self.get_val_item(idx)
        elif self.split == 'test':
            return self.get_test_item(idx)
        else:
            return self.get_train_item(idx)
