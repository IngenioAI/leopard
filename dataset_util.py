import os
import json

'''
dataset_example = [
    {
        "name": "CelebA",
        "type": "Image",
        "storageId": "2",
        "storagePath": "/CelebA/Img/img_align_celeba",
        "description": "CelebA Face image dataset"
    }
]
'''

current_dataset_list = None

def get_dataset_info():
    global current_dataset_list
    if current_dataset_list is None:
        dataset_info_path = os.path.join("config", "dataset.json")
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, "rt", encoding="utf-8") as fp:
                current_dataset_list = json.load(fp)
        else:
            current_dataset_list = []

    return current_dataset_list

def save_dataset_info(info):
    global current_dataset_list
    current_dataset_list = info
    dataset_info_path = os.path.join("config", "dataset.json")
    with open(dataset_info_path, "wt", encoding="utf-8") as fp:
        json.dump(info, fp)
    return True