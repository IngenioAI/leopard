from math import ceil
from functools import partial
from random import random, choice
from io import BytesIO
import os
import sys

import torch
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision.transforms as T


from pytorch_kid34k.data_utils.custom_dataset import TestDataset_torchvision, MultiDataset_torchvision, BinaryDataset_torchvision, BinaryTestDataset_torchvision


def get_dataloader_torchvision(
    json_path, 
    batch_size,
    val_batch_size, 
    img_H,
    img_W, 
    fold, 
    augmentation_config, 
    augmentation_test_config, 
    use_cutout=False, 
    cutout_n_holes=None,
    cutout_length=None,
    n_fold=5, 
    n_workers=4, 
    use_DDP=False, 
    n_gpus=1,
    use_3class=False,
    except_class=None,
    use_balsampler=False
):
    train_transform = getattr(torchvision_aug(img_H, img_W), f"get_aug{augmentation_config}")()
    test_transform = getattr(torchvision_test_aug(img_H, img_W), f"get_aug{augmentation_test_config}")()
    if use_cutout:
        train_transform.transforms.append(Cutout(n_holes=cutout_n_holes, length=cutout_length))
    if except_class is None:
        train_dataset = MultiDataset_torchvision(json_path, transform=train_transform, fold=fold, n_fold=n_fold, is_train=True, is_valid=False, use_3class=use_3class)
        valid_dataset = MultiDataset_torchvision(json_path, transform=test_transform, fold=fold, n_fold=n_fold, is_train=True, is_valid=True, use_3class=use_3class)
    else:
        train_dataset = BinaryDataset_torchvision(json_path, transform=train_transform, is_valid=False, except_class=except_class)
        valid_dataset = BinaryDataset_torchvision(json_path, transform=test_transform, is_valid=True, except_class=except_class)
    if use_DDP:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    if not use_DDP and use_balsampler:
        from torchsampler import ImbalancedDatasetSampler
        train_sampler = ImbalancedDatasetSampler(train_dataset)
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=max(batch_size//n_gpus, 1), sampler=train_sampler, shuffle=shuffle, num_workers=n_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
    return train_loader, valid_loader    

#for MIA
def get_dataloader_torchvision_ver2(
    json_path, 
    batch_size,
    val_batch_size, 
    img_H,
    img_W, 
    fold, 
    augmentation_config, 
    augmentation_test_config, 
    use_cutout=False, 
    cutout_n_holes=None,
    cutout_length=None,
    n_fold=5, 
    n_workers=4, 
    use_DDP=False, 
    n_gpus=1,
    use_3class=False,
    except_class=None,
    use_balsampler=False
):
    train_transform = getattr(torchvision_aug(img_H, img_W), f"get_aug{augmentation_config}")()
    if use_cutout:
        train_transform.transforms.append(Cutout(n_holes=cutout_n_holes, length=cutout_length))
    if except_class is None:
        train_dataset = MultiDataset_torchvision(json_path, transform=train_transform, fold=fold, n_fold=n_fold, is_train=True, is_valid=False, use_3class=use_3class)
    else:
        train_dataset = BinaryDataset_torchvision(json_path, transform=train_transform, is_valid=False, except_class=except_class)
    if use_DDP:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    if not use_DDP and use_balsampler:
        from torchsampler import ImbalancedDatasetSampler
        train_sampler = ImbalancedDatasetSampler(train_dataset)
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=max(batch_size//n_gpus, 1), sampler=train_sampler, shuffle=shuffle, num_workers=n_workers, pin_memory=True)
    return train_loader    


def get_test_dataloader_torchvision(
    json_path, 
    test_batch_size, 
    img_H, 
    img_W, 
    augmentation_test_config, 
    n_workers=4, 
    use_3class=False,
    except_class=None
):
    test_transform = getattr(torchvision_test_aug(img_H, img_W), f"get_aug{augmentation_test_config}")()
    if except_class is None:
        test_dataset = TestDataset_torchvision(json_path, transform=test_transform, use_3class=use_3class)
    else:
        test_dataset = BinaryTestDataset_torchvision(json_path, transform=test_transform, except_class=except_class)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
    return test_loader

# regularization technique. https://github.com/fitushar/Improved-Regularization-of-Convolutional-Neural-Networks 
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img
    
class torchvision_aug():
    def __init__(self, img_H, img_W):
        self.img_H = img_H
        self.img_W = img_W
        self.jpeg_dict = {'cv2': self.cv2_jpg, 'pil': self.pil_jpg}
    @staticmethod
    def gaussian_blur(img, sigma):
        gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
        gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
        gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
    @staticmethod
    def sample_continuous( s):
        if len(s) == 1:
            return s[0]
        if len(s) == 2:
            rg = s[1] - s[0]
            return random() * rg + s[0]
        raise ValueError("Length of iterable s should be 1 or 2.")
    @staticmethod
    def sample_discrete(s):
        if len(s) == 1:
            return s[0]
        return choice(s)
    @staticmethod
    def cv2_jpg(img, compress_val):
        img_cv2 = img[:,:,::-1]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
        result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg[:,:,::-1]
    @staticmethod
    def pil_jpg(img, compress_val):
        out = BytesIO()
        img = Image.fromarray(img)
        img.save(out, format='jpeg', quality=compress_val)
        img = Image.open(out)
        # load from memory before ByteIO closes
        img = np.array(img)
        out.close()
        return img
    def jpeg_from_key(self, img, compress_val, key):
        method = self.jpeg_dict[key]
        return method(img, compress_val)
    def gaublur_jpeg(self, img):
        img = np.array(img)
        if random() < 0.1:
            sig = self.sample_continuous([0.0, 3.0])
            self.gaussian_blur(img, sig)
        if random() < 0.1:
            method = self.sample_discrete(["cv2", "pil"])
            qual = self.sample_discrete([30, 100])
            img = self.jpeg_from_key(img, qual, method)
        return Image.fromarray(img)
    def pad_to_minimum_size(self, image):
        w, h = image.size
        h_diff = h - self.img_H
        w_diff = w - self.img_W
        h_pad = ceil(abs(h_diff) / 2) if h_diff < 0 else 0
        w_pad = ceil(abs(w_diff) / 2) if w_diff < 0 else 0
        if h_pad == 0 and w_pad == 0:
            return image
        else:
            return T.functional.pad(image, [h_pad, w_pad])

    def rotate_align_pad_to_minimum_size(self, image):
        w, h = image.size
        if w < h:
            image = image.rotate(90, expand=1)
        w, h = image.size       
        h_diff = h - self.img_H
        w_diff = w - self.img_W
        h_pad = ceil(abs(h_diff) / 2) if h_diff < 0 else 0
        w_pad = ceil(abs(w_diff) / 2) if w_diff < 0 else 0
        if h_pad == 0 and w_pad == 0:
            return image
        else:
            return T.functional.pad(image, [h_pad, w_pad])
    def rotate(self, image):
        w, h = image.size
        if w<h:
            image = image.rotate(90, expand=1)
        return image        

    def get_aug19(self):
        transform = T.Compose([
            T.Lambda(self.rotate_align_pad_to_minimum_size),
            T.RandomResizedCrop((self.img_H, self.img_W), scale=(0.5,1),ratio=(0.9,1)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform
   
class torchvision_test_aug():
    def __init__(self, img_H, img_W):
        self.img_H = img_H
        self.img_W = img_W
        self.jpeg_dict = {'cv2': self.cv2_jpg, 'pil': self.pil_jpg}
    @staticmethod
    def gaussian_blur(img, sigma):
        gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
        gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
        gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
    @staticmethod
    def sample_continuous( s):
        if len(s) == 1:
            return s[0]
        if len(s) == 2:
            rg = s[1] - s[0]
            return random() * rg + s[0]
        raise ValueError("Length of iterable s should be 1 or 2.")
    @staticmethod
    def sample_discrete(s):
        if len(s) == 1:
            return s[0]
        return choice(s)
    @staticmethod
    def cv2_jpg(img, compress_val):
        img_cv2 = img[:,:,::-1]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
        result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg[:,:,::-1]
    @staticmethod
    def pil_jpg(img, compress_val):
        out = BytesIO()
        img = Image.fromarray(img)
        img.save(out, format='jpeg', quality=compress_val)
        img = Image.open(out)
        # load from memory before ByteIO closes
        img = np.array(img)
        out.close()
        return img
    def jpeg_from_key(self, img, compress_val, key):
        method = self.jpeg_dict[key]
        return method(img, compress_val)
    def gaublur_jpeg(self, img):
        img = np.array(img)
        if random() < 0.1:
            sig = self.sample_continuous([0.0, 3.0])
            self.gaussian_blur(img, sig)
        if random() < 0.1:
            method = self.sample_discrete(["cv2", "pil"])
            qual = self.sample_discrete([30, 100])
            img = self.jpeg_from_key(img, qual, method)
        return Image.fromarray(img)
    def pad_to_minimum_size(self, image):
        w, h= image.size
        h_diff = h - self.img_H
        w_diff = w - self.img_W
        h_pad = ceil(abs(h_diff) / 2) if h_diff < 0 else 0
        w_pad = ceil(abs(w_diff) / 2) if w_diff < 0 else 0
        if h_pad == 0 and w_pad == 0:
            return image
        else:
            return T.functional.pad(image, [h_pad, w_pad])
    def rotate_align_pad_to_minimum_size(self, image):
        w, h = image.size
        if w < h:
            image = image.rotate(90, expand=1)
        w, h = image.size       
        h_diff = h - self.img_H
        w_diff = w - self.img_W
        h_pad = ceil(abs(h_diff) / 2) if h_diff < 0 else 0
        w_pad = ceil(abs(w_diff) / 2) if w_diff < 0 else 0
        if h_pad == 0 and w_pad == 0:
            return image
        else:
            return T.functional.pad(image, [h_pad, w_pad])
    def rotate(self, image):
        w, h = image.size
        if w<h:
            image = image.rotate(90, expand=1)
        return image  
   
    def get_aug5(self):
        transform = T.Compose([
            T.Lambda(self.rotate_align_pad_to_minimum_size),
            T.Resize((self.img_H, self.img_W)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform
    