import json
import os

from PIL import Image
import cv2
from torch.utils.data import Dataset

class MultiDataset_torchvision(Dataset):
    def __init__(self, json_path, transform, fold, n_fold=5, is_train=True, is_valid=False, use_3class=False):
        super().__init__()
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        if not is_valid:
            r_ps = json_file["real_train"]
            f_ps = json_file["fake_train"]
        else:
            r_ps = json_file["real_valid"]
            f_ps = json_file["fake_valid"]
        real_labels = [0 for _ in range(len(r_ps))]
        fake_labels = []
        fake_classes = []
        n_screen = 0
        n_print = 0
        for f in f_ps:
            if "screen" in f:
                fake_labels.append(1)
                fake_classes.append(1)
                n_screen += 1
            elif "print" in f: 
                n_print += 1
                fake_classes.append(2)
                if not use_3class:
                    fake_labels.append(1)
                else:
                    fake_labels.append(2)
        assert len(fake_labels) == len(f_ps), f"screen : {n_screen}, print : {n_print}, f_ps : {len(f_ps)}"

        
        self.print_len = n_print
        self.screen_len = n_screen
        self.genuine_len = len(r_ps)

        self.img_paths = r_ps + f_ps
        self.labels = real_labels + fake_labels
        self.classes = real_labels + fake_classes
        assert len(self.img_paths) == len(self.labels) == len(self.classes)
        self.transform = transform
    def get_n_paths(self):
        return f"# of paths : {len(self.img_paths)}"
    def sample_label_count(self):
        return f"# of print : {self.print_len}\n# of screen : {self.screen_len}\n# of genuine : {self.genuine_len}"


    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        # train_val, fake train : /home/yujeong/project/privacy/pe2/target/cheapfake/data/splitted_ver2/ID_113_U04/screen/ID_113#screen#336#20221110163434_cropImage_w829_h488.jpeg
        # train_val, real train: "/home/yujeong/project/privacy/pe2/target/cheapfake/data/splitted_ver2/DL_024_U37/genuine/DL_024#genuine#303#20230121121135_Android_Crop_image.jpg",
        # try:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        class_ = self.classes[idx]
        img = self.transform(img)
        # except OSError as e:
        #     print(f"Error opening image at index {idx}: {e}")
        #     print(f"Problematic image path: {self.img_paths[idx]}")
        #     return self.__getitem__(idx + 1)
        return img, label, class_
    
class BinaryDataset_torchvision(Dataset):
    def __init__(self, json_path, transform, is_valid=False, except_class=None):
        super().__init__()
        assert except_class in ["real", "screen", "print"]
        with open(json_path, "r") as f:
            json_file = json.load(f)
        if not is_valid:
            r_ps = json_file["real_train"]
            f_ps = json_file["fake_train"]
        else:
            r_ps = json_file["real_valid"]
            f_ps = json_file["fake_valid"]
        real_paths = r_ps
        screen_paths = []
        print_paths = []
        for f in f_ps:
            if "screen" in f: screen_paths.append(f)
            elif "print" in f: print_paths.append(f)
        if except_class == "real":
            class0 = screen_paths
            class1 = print_paths
        elif except_class == "screen":
            class0 = real_paths
            class1 = print_paths
        else:
            class0 = real_paths
            class1 = screen_paths
        self.img_paths = class0 + class1
        self.labels = [0 for _ in range(len(class0))] + [1 for _ in range(len(class1))]
        self.classes = [0 for _ in range(len(class0))] + [1 for _ in range(len(class1))]
        assert len(self.img_paths) == len(self.labels) == len(self.classes)
        self.transform = transform
    def get_n_paths(self):
        return f"# of paths : {len(self.img_paths)}"
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        # try:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        class_ = self.classes[idx]
        img = self.transform(img)
        # except OSError as e:
        #     print(f"Error opening image at index {idx}: {e}")
        #     print(f"Problematic image path: {self.img_paths[idx]}")
        #     return self.__getitem__(idx + 1)
        
        return img, label, class_
    
class TestDataset_torchvision(Dataset):
    def __init__(self, json_path, transform, use_3class):
        super().__init__()
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        self.img_paths = json_file["real"] + json_file["screen"] + json_file["print"]
        if not use_3class:
            self.labels = [0 for _ in range(len(json_file["real"]))] + [1 for _ in range(len(json_file["screen"]))] + [1 for _ in range(len(json_file["print"]))]
        else:
            self.labels = [0 for _ in range(len(json_file["real"]))] + [1 for _ in range(len(json_file["screen"]))] + [2 for _ in range(len(json_file["print"]))]
        self.classes = [0 for _ in range(len(json_file["real"]))] + [1 for _ in range(len(json_file["screen"]))] + [2 for _ in range(len(json_file["print"]))]
        assert len(self.img_paths) == len(self.labels) == len(self.classes)
        self.transform = transform


        self.print_len = len(json_file["real"])
        self.screen_len = len(json_file["screen"])
        self.genuine_len = len(json_file["print"])


    def get_n_paths(self):
        return f"# of paths : {len(self.img_paths)}"

    def sample_label_count(self):
        return f"# of print : {self.print_len}\n# of screen : {self.screen_len}\n# of genuine : {self.genuine_len}"
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        # try:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        class_ = self.classes[idx]
        img = self.transform(img)
        # except OSError as e:
        #     print(f"Error opening image at index {idx}: {e}")
        #     print(f"Problematic image path: {self.img_paths[idx]}")
        #     return self.__getitem__(idx + 1)
        return img, label, class_
    
class BinaryTestDataset_torchvision(Dataset):
    def __init__(self, json_path, transform, except_class):
        super().__init__()
        assert except_class in ["real", "screen", "print"]
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        real_paths = json_file["real"]
        screen_paths = json_file["screen"]
        print_paths = json_file["print"]
        if except_class == "real":
            class0 = screen_paths
            class1 = print_paths
        elif except_class == "screen":
            class0 = real_paths
            class1 = print_paths
        else:
            class0 = real_paths
            class1 = screen_paths
        self.img_paths = class0 + class1
        self.labels = [0 for _ in range(len(class0))] + [1 for _ in range(len(class1))]
        self.classes = [0 for _ in range(len(class0))] + [1 for _ in range(len(class1))]
        self.transform = transform
    def get_n_paths(self):
        return f"# of paths : {len(self.img_paths)}"
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        # try:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        class_ = self.classes[idx]
        img = self.transform(img)
        # except OSError as e:
        #     print(f"Error opening image at index {idx}: {e}")
        #     print(f"Problematic image path: {self.img_paths[idx]}")
        #     return self.__getitem__(idx + 1)
        
        return img, label, class_

# except print label
class MultiDataset_torchvision_test(Dataset):
    def __init__(self, json_path, sample_label, transform, fold, n_fold=5, is_train=True, is_valid=False, use_3class=False):
        super().__init__()
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        if not is_valid:
            r_ps = json_file["real_train"]
            f_ps = json_file["fake_train"]
        else:
            r_ps = json_file["real_valid"]
            f_ps = json_file["fake_valid"]
        
        real_labels = [0 for _ in range(len(r_ps))]
        fake_labels = []
        fake_classes = []
        screen_paths = []
        n_target = 0
        n_other = 0

        if sample_label == "screen":
            target_label = "screen"
            other_label = "print"
        else:
            target_label = "print"
            other_label = "screen"

        for f in f_ps:
            if target_label in f:
                fake_labels.append(1)
                fake_classes.append(1)
                screen_paths.append(f)
                n_target += 1
                
            elif other_label in f: 
                n_other += 1

        total_size = len(f_ps) - n_other

        if sample_label == "screen":
            self.print_len = n_other
            self.screen_len = n_target
        else:
            self.print_len = n_target
            self.screen_len = n_other

        self.genuine_len = len(r_ps)

        assert len(fake_labels) == total_size, f"{n_target} : {n_target}, f_ps : {len(f_ps)}"
        self.img_paths = r_ps + screen_paths
        self.labels = real_labels + fake_labels
        self.classes = real_labels + fake_classes
        assert len(self.img_paths) == len(self.labels) == len(self.classes)
        self.transform = transform

    def get_n_paths(self):
        return f"# of paths : {len(self.img_paths)}"
    def sample_label_count(self):
        return f"# of print : {self.print_len}\n# of screen : {self.screen_len}\n# of genuine : {self.genuine_len}"

    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        # train_val, fake train : /home/yujeong/project/privacy/pe2/target/cheapfake/data/splitted_ver2/ID_113_U04/screen/ID_113#screen#336#20221110163434_cropImage_w829_h488.jpeg
        # train_val, real train: "/home/yujeong/project/privacy/pe2/target/cheapfake/data/splitted_ver2/DL_024_U37/genuine/DL_024#genuine#303#20230121121135_Android_Crop_image.jpg",
        # try:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        class_ = self.classes[idx]
        img = self.transform(img)
        # except OSError as e:
        #     print(f"Error opening image at index {idx}: {e}")
        #     print(f"Problematic image path: {self.img_paths[idx]}")
        #     return self.__getitem__(idx + 1)
        return img, label, class_
    
