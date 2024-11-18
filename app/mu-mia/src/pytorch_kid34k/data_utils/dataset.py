
import pandas as pd
import random 
import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from pytorch_kid34k.data_utils.custom_dataset import TestDataset_torchvision, MultiDataset_torchvision, BinaryDataset_torchvision, BinaryTestDataset_torchvision, MultiDataset_torchvision_test
from pytorch_kid34k.data_utils.dataloader_ import torchvision_aug, Cutout, torchvision_test_aug


def get_dataset(
        train_data_json_path,
        test_data_json_path,
        print_test=False,
        screen_test=False,
        small_test=False,
        img_H=512, 
        img_W=800,
        fold=1, 
        augmentation_config=19,
        augmentation_test_config=5,
        use_cutout=False,
        cutout_n_holes=1,
        cutout_length=200,
        n_fold=1,
        use_3class=False,
        except_class=None):
    
    train_transform = getattr(torchvision_aug(img_H, img_W), f"get_aug{augmentation_config}")()
    test_transform = getattr(torchvision_test_aug(img_H, img_W), f"get_aug{augmentation_test_config}")()
    if use_cutout:
        train_transform.transforms.append(Cutout(n_holes=cutout_n_holes, length=cutout_length))
    if except_class is None:
        if print_test:
            train_dataset = MultiDataset_torchvision_test(train_data_json_path, "print", transform=train_transform, fold=fold, n_fold=n_fold, is_train=True, is_valid=False, use_3class=use_3class)
            valid_dataset = MultiDataset_torchvision_test(train_data_json_path, "print", transform=test_transform, fold=fold, n_fold=n_fold, is_train=True, is_valid=True, use_3class=use_3class)
            test_dataset = TestDataset_torchvision(test_data_json_path, transform=test_transform, use_3class=use_3class)
        elif screen_test:
            train_dataset = MultiDataset_torchvision_test(train_data_json_path, "screen", transform=train_transform, fold=fold, n_fold=n_fold, is_train=True, is_valid=False, use_3class=use_3class)
            valid_dataset = MultiDataset_torchvision_test(train_data_json_path, "screen", transform=test_transform, fold=fold, n_fold=n_fold, is_train=True, is_valid=True, use_3class=use_3class)
            test_dataset = TestDataset_torchvision(test_data_json_path, transform=test_transform, use_3class=use_3class)
        else:
            train_dataset = MultiDataset_torchvision(train_data_json_path, transform=train_transform, fold=fold, n_fold=n_fold, is_train=True, is_valid=False, use_3class=use_3class)
            valid_dataset = MultiDataset_torchvision(train_data_json_path, transform=test_transform, fold=fold, n_fold=n_fold, is_train=True, is_valid=True, use_3class=use_3class)
            test_dataset = TestDataset_torchvision(test_data_json_path, transform=test_transform, use_3class=use_3class)
    else:
        train_dataset = BinaryDataset_torchvision(train_data_json_path, transform=train_transform, is_valid=False, except_class=except_class)
        valid_dataset = BinaryDataset_torchvision(train_data_json_path, transform=test_transform, is_valid=True, except_class=except_class)
        test_dataset = BinaryTestDataset_torchvision(test_data_json_path, transform=test_transform, except_class=except_class)

    if small_test:
        train_size = len(train_dataset) // 5
        valid_size = len(valid_dataset) // 5
        test_size = len(test_dataset) // 5

        train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
        valid_dataset = torch.utils.data.Subset(valid_dataset, range(valid_size))
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_size))


    return train_dataset, valid_dataset, test_dataset