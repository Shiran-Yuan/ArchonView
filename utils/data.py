import os
import os.path as osp

import PIL.Image as PImage
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, transforms

import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import random

class CLIPDataset(Dataset):
    def __init__(self, root, loader, transform, train):
        self.root = osp.join(root, 'train' if train else 'val')
        self.loader = loader
        self.transform = transform
        self.subdirs = [osp.join(self.root, d) for d in os.listdir(self.root)]
        self.rng = np.random.default_rng(seed=42)
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2])
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])
    def get_T(self, target_RT, cond_RT):
        R1, T1 = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R1.T @ T1
        R2, T2= cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R2.T @ T2
        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T
    def __len__(self): return len(self.subdirs)
    def __getitem__(self, idx):
        path = self.subdirs[idx]
        num1 = self.rng.integers(0, 12)
        num2 = self.rng.integers(0, 12)
        tar1 = str(num1).zfill(3)
        tar2 = str(num2).zfill(3)
        source = self.loader(osp.join(path, f'{tar1}.png'))
        target = self.loader(osp.join(path, f'{tar2}.png'))
        with open(osp.join(path, f'{tar1}.npy'), 'rb') as fs:
            with open(osp.join(path, f'{tar2}.npy'), 'rb') as ft:
                pose_s, pose_t = np.load(fs), np.load(ft)
        with open(osp.join(path, f'{tar1}e.npy'), 'rb') as f:
            emb = torch.tensor(np.load(f))
        d_T = self.get_T(pose_t, pose_s)
        return self.transform(source), self.transform(target), d_T.view(-1), emb.view(-1)

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

    
def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # build dataset
    train_set = CLIPDataset(root=data_path, loader=pil_loader, transform=train_aug, train=True)
    val_set = CLIPDataset(root=data_path, loader=pil_loader, transform=val_aug, train=False)
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
    return train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
