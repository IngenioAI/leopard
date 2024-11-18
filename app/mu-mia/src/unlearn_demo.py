import os
import random
import numpy as np
import argparse
from copy import deepcopy
from time import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from torchvision import transforms
from collections import defaultdict
from facenet_pytorch import InceptionResnetV1
import json
from torch.utils.data import Dataset
from PIL import Image

# def print_conf(conf):
#     print("\n#### configurations ####")
#     for k, v in vars(conf).items():
#             print('{}: {}'.format(k, v))
#     print("########################\n")

# def fix_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if use multi-GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)
#     random.seed(seed)

class CustomImgDataset(Dataset):
    def __init__(self, conf, data_file, transform=None):
        with open(data_file, 'r') as f:
            self.image_paths = json.load(f)
        self.conf = conf
        self.transform = transform
        self.class_to_idx = {folder: idx for idx, folder in enumerate(sorted(set(os.path.basename(os.path.dirname(path)) for path in self.image_paths)))}
        self.samples = [(os.path.join(self.conf.data_path, path), self.class_to_idx[os.path.basename(os.path.dirname(path))]) for path in self.image_paths]
        self.classes = list(self.class_to_idx.keys())
        self.targets = [sample[1] for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
def get_dataset_demo(conf, trans=None):
    if trans is None:
        trans = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=conf.mean, std=conf.std)
            ])
    
    shadow_train_file = f'{conf.data_path}/shadow_train_data_info.json'
    shadow_test_file = f'{conf.data_path}/shadow_test_data_info.json'
    target_train_file = f'{conf.data_path}/target_train_data_info.json'
    target_test_file = f'{conf.data_path}/target_test_data_info.json'

    shadow_trainset = CustomImgDataset(conf, shadow_train_file, transform=trans)
    shadow_testset = CustomImgDataset(conf, shadow_test_file, transform=trans)
    target_trainset = CustomImgDataset(conf, target_train_file, transform=trans)
    target_testset = CustomImgDataset(conf, target_test_file, transform=trans)


    return shadow_trainset, shadow_testset, target_trainset, target_testset

def get_dataloader(train_dataset, test_dataset, conf):
    train_loader = DataLoader(dataset=train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def split_class_data(dataset, forget_class_idx):
    targets = torch.tensor(dataset.targets)
    forget_indices = (targets == forget_class_idx).nonzero(as_tuple=True)[0].tolist()
    remain_indices = (targets != forget_class_idx).nonzero(as_tuple=True)[0].tolist()
    return forget_indices, remain_indices

def get_unlearn_loader(train_set, test_set, conf):
    indices_path = os.path.join(conf.data_path, 'dataset_indices')
    train_indices_path = os.path.join(indices_path, f'train_indices_{conf.forget_class_idx}.pt')
    test_indices_path = os.path.join(indices_path, f'test_indices_{conf.forget_class_idx}.pt')
    if os.path.exists(train_indices_path) and os.path.exists(test_indices_path):
        print(f"Load indices from {train_indices_path} and {test_indices_path}")
        train_indices = torch.load(train_indices_path)
        test_indices = torch.load(test_indices_path)
        train_forget_indices = train_indices['forget']
        train_remain_indices = train_indices['remain']
        test_forget_indices = test_indices['forget']
        test_remain_indices = test_indices['remain']
    else:
        train_forget_indices, train_remain_indices = split_class_data(train_set, conf.forget_class_idx)
        test_forget_indices, test_remain_indices = split_class_data(test_set, conf.forget_class_idx)
        train_indices = {'forget': train_forget_indices, 'remain': train_remain_indices}
        test_indices = {'forget': test_forget_indices, 'remain': test_remain_indices}
        os.makedirs(indices_path, exist_ok=True)
        torch.save(train_indices, train_indices_path)
        torch.save(test_indices, test_indices_path)
    
    train_forget_set = Subset(train_set, train_forget_indices)
    train_retain_set = Subset(train_set, train_remain_indices)

    test_forget_set = Subset(test_set, test_forget_indices)
    test_retain_set = Subset(test_set, test_remain_indices)

    train_forget_loader = DataLoader(dataset=train_forget_set, batch_size=conf.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_retain_loader = DataLoader(dataset=train_retain_set, batch_size=conf.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_forget_loader = DataLoader(dataset=test_forget_set, batch_size=conf.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_retain_loader = DataLoader(dataset=test_retain_set, batch_size=conf.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_forget_loader, train_retain_loader, test_forget_loader, test_retain_loader

def train(model, dataloader, criterion, optimizer, conf):
    model.train()
    correct, losses, total=0, 0, 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        images, labels = images.to(conf.device), labels.to(conf.device)
        outputs = model(images)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        losses += loss.item() * len(labels)
        total += labels.size(0)
    return correct / total, losses / total

@torch.no_grad()
def evaluate(model, dataloader, criterion, conf):
    model.eval()
    correct, losses, total = 0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(conf.device), labels.to(conf.device)
        outputs = model(images)
        loss=criterion(outputs, labels)

        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        losses += loss.item() * len(labels)
        total += labels.size(0)

    return correct / total, losses / total

class PGD():
    def __init__(self, model=None, eps=8, alpha=2, iters=10, denorm=True):
        self.model = model
        self.eps = eps/255
        self.alpha = alpha/255
        self.iters = iters
        self.denorm = denorm
        self.device = next(model.parameters()).device

    def set_normalization(self, mean, std):
        n_channels = len(mean)
        self.mean = torch.tensor(mean, device=self.device).reshape(1, n_channels, 1, 1)
        self.std = torch.tensor(std, device=self.device).reshape(1, n_channels, 1, 1)

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, images, labels, target_labels=None):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if target_labels is not None:
            target_labels = target_labels.clone().detach().to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.iters):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            if target_labels is not None:
                loss = -criterion(outputs, target_labels)
            else:
                loss = criterion(outputs, labels)
            grad_sign = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad_sign.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images.detach()
    
    def __call__(self, images, labels, target_labels=None, return_adv_labels=False):
        self.model.eval()
        if self.denorm:
            images = self.denormalize(images)
            adv_inputs = self.forward(images, labels, target_labels)
            adv_inputs = self.normalize(adv_inputs)
        else:
            adv_inputs = self.forward(images, labels, target_labels)

        if return_adv_labels:
            with torch.no_grad():
                adv_labels = self.model(adv_inputs.to(self.device)).argmax(dim=1)
            self.model.train()
            return adv_inputs.detach().cpu(), adv_labels.detach().cpu()
        self.model.train()
        return adv_inputs

def Unlearn(original_model, train_forget_loader, conf):
    original_model = original_model.to(conf.device)
    unlearn_model = deepcopy(original_model).to(conf.device)
    attack = PGD(model=original_model.to(conf.device), eps=conf.eps, alpha=conf.alpha, iters=conf.iters, denorm=conf.denorm)
    attack.set_normalization(conf.mean, conf.std)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=conf.unlearn_lr, momentum=conf.unlearn_momentum, weight_decay=conf.unlearn_weight_decay)
    
    for epoch in range(conf.unlearn_epochs):
        original_model.train()
        start = time()
        nearest_label = []
        correct, nums_filpped, total, losses = 0, 0, 0, 0
        unlearn_model.train()
        for images, labels in train_forget_loader:
            images = images.to(conf.device)
            _, adv_labels = attack(images, labels, return_adv_labels=True)

            nearest_label.append(adv_labels.tolist())
            nums_filpped += (labels != adv_labels).sum().item()

            outputs = unlearn_model(images)
            loss = criterion(outputs, adv_labels.to(conf.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(dim=1, keepdim=True).detach().cpu()
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
            losses += loss.item() * labels.size(0)
        torch.cuda.synchronize()
        print(f'Epoch {epoch}|Time {time()-start:.3f}|Loss {losses / total:.4f}|Acc {correct/total*100:.4f}|Flipped {nums_filpped/total:.4f}')
    return unlearn_model

# def build_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--mode", type=str, default='original', choices=['original', 'retrain', 'unlearn'])
#     #### dataset ####
#     parser.add_argument("--dataset", type=str, default='VGGFace2')
#     parser.add_argument("--data_path", type=str, default='./data')
#     parser.add_argument("--batch_size", type=int, default=128)
#     parser.add_argument("--mean", type=list, default=[0.5, 0.5, 0.5])
#     parser.add_argument("--std", type=list, default=[0.5, 0.5, 0.5])
#     #### train & test ####
#     parser.add_argument("--epochs", type=int, default=8)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--momentum", type=float, default=0.9)
#     parser.add_argument("--weight_decay", type=float, default=1e-4)
#     #### save & load ####
#     parser.add_argument("--evaluate", default=True)
#     parser.add_argument("--save_dir", type=str, default='./')
#     #### for unlearn ####
#     parser.add_argument("--model_load_path", default='./VGGFace2_original_model.pth')
#     parser.add_argument("--forget_class_idx", type=int, default=3)
#     parser.add_argument("--eps", type=float, default=32)
#     parser.add_argument("--alpha", type=float, default=1)
#     parser.add_argument("--iters", type=int, default=10)
#     parser.add_argument("--denorm", default=True)
#     parser.add_argument("--unlearn_epochs", type=int, default=5)
#     parser.add_argument("--unlearn_lr", type=float, default=6e-4)
#     parser.add_argument("--unlearn_momentum", type=float, default=0.9)
#     parser.add_argument("--unlearn_weight_decay", type=float, default=1e-4)
#     args = parser.parse_args()
#     args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     return args

def main(conf):
    print(f'Setting {conf.dataset} dataset...\n')
    shadow_trainset, shadow_testset, target_trainset, target_testset = get_dataset_demo(conf)
    sh_train_loader, sh_test_loader = get_dataloader(shadow_trainset, shadow_testset, conf)
    trg_train_loader, trg_test_loader = get_dataloader(target_trainset, target_testset, conf)

    trg_train_forget_loader, trg_train_retain_loader, trg_test_forget_loader, trg_test_retain_loader = get_unlearn_loader(target_trainset, target_testset, conf)

    def get_class_counts(dataset):
        class_counts = defaultdict(int)
        for _, label in dataset.samples:
            class_counts[dataset.classes[label]] += 1
        return class_counts

    sh_train_counts = get_class_counts(shadow_trainset)
    sh_test_counts = get_class_counts(shadow_testset)
    trg_train_counts = get_class_counts(target_trainset)
    trg_test_counts = get_class_counts(target_testset)
    for class_name in sh_train_counts.keys():
        print(f"{class_name}: Target Train={trg_train_counts.get(class_name, 0)}, Target Test={trg_test_counts.get(class_name, 0)}")
        print(f"{class_name}: Shadow Train={sh_train_counts.get(class_name, 0)}, Shadow Test={sh_test_counts.get(class_name, 0)}")

    print(f'\nBuilding Model...\n')
    sh_model = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(shadow_trainset.class_to_idx)
    ).to(conf.device)

    trg_model = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(target_trainset.class_to_idx)
    ).to(conf.device)

    loss_fn = torch.nn.CrossEntropyLoss()

    trg_save_path = f"{conf.save_dir}/target"
    sh_save_path = f"{conf.save_dir}/shadow"
    os.makedirs(trg_save_path, exist_ok = True)
    os.makedirs(sh_save_path, exist_ok = True)

    if conf.mode in ['original', 'retrain']:
        total_start = time()
        print(f'Start training {conf.mode.capitalize()} model...')
        if conf.mode == 'retrain':
            trg_train_loader = trg_train_retain_loader
            trg_test_loader = trg_test_retain_loader

        print("Target Model Training")
        optimizer = optim.SGD(trg_model.parameters(), lr=conf.lr, momentum=conf.momentum, weight_decay=conf.weight_decay)
        for i in tqdm(range(conf.org_epochs), desc='Training'):
            train_acc, train_loss = train(trg_model, trg_train_loader, loss_fn, optimizer, conf)
            print(f'EPOCH {i}/{conf.org_epochs} : Target Train loss: {train_loss:.4f}')
        torch.cuda.synchronize()

        print(f'Training time: {time()-total_start:.3f}\n')

        test_acc, test_loss = evaluate(trg_model, trg_test_loader, loss_fn, conf)
        print(f'Target Test Acc: {test_acc*100:.4f}|Target Test Loss: {test_loss:.4f}')

        print("Shadow Model Training")
        optimizer = optim.SGD(sh_model.parameters(), lr=conf.lr, momentum=conf.momentum, weight_decay=conf.weight_decay)
        for i in tqdm(range(conf.org_epochs), desc='Training'):
            train_acc, train_loss = train(sh_model, sh_train_loader, loss_fn, optimizer, conf)
            print(f'EPOCH {i}/{conf.org_epochs} : Shadow Train loss: {train_loss:.4f}')
        torch.cuda.synchronize()

        print(f'Training time: {time()-total_start:.3f}\n')

        test_acc, test_loss = evaluate(sh_model, sh_test_loader, loss_fn, conf)
        print(f'Shadow Test Acc: {test_acc*100:.4f}|Shadow Test Loss: {test_loss:.4f}')

        if conf.evaluate:
            trg_train_forget_acc, trg_train_forget_loss = evaluate(trg_model, trg_train_forget_loader, loss_fn, conf)
            trg_train_retain_acc, trg_train_forget_loss = evaluate(trg_model, trg_train_retain_loader, loss_fn, conf)
            trg_test_forget_acc, trg_test_forget_loss = evaluate(trg_model, trg_test_forget_loader, loss_fn, conf)
            trg_test_retain_acc, trg_train_retain_loss = evaluate(trg_model, trg_test_retain_loader, loss_fn, conf)
            print(f'Target Train Forget Acc: {trg_train_forget_acc*100:.4f}|Target Train Retain Acc: {trg_train_retain_acc*100:.4f}|Target Test Forget Acc: {trg_test_forget_acc*100:.4f}|Target Test Retain Acc: {trg_test_retain_acc*100:.4f}\n')
            with open(os.path.join(conf.save_dir, f'{conf.dataset}_{conf.mode}_target_result_forget{conf.forget_class_idx}.txt'), 'w') as f:
                f.write(f'forget_class_idx: {conf.forget_class_idx}|unlearn_epoch: {conf.unlearn_epochs}|unlearn_lr: {conf.unlearn_lr}\n')
                f.write(f'Target Train Forget Acc: {trg_train_forget_acc*100:.4f}|Target Train Retain Acc: {trg_train_retain_acc*100:.4f}|Target Test Forget Acc: {trg_test_forget_acc*100:.4f}|Target Test Retain Acc: {trg_test_retain_acc*100:.4f}\n')

        print('Saving model...\n')
        print(conf.mode)
        if conf.mode == 'original':
            torch.save(trg_model.state_dict(), os.path.join(trg_save_path, f'{conf.dataset}_{conf.mode}_target_model.pth'))
            torch.save(sh_model.state_dict(), os.path.join(sh_save_path, f'{conf.dataset}_{conf.mode}_shadow_model.pth'))
        else:
            torch.save(trg_model.state_dict(), os.path.join(trg_save_path, f'{conf.dataset}_{conf.mode}_target_model_forget{conf.forget_class_idx}.pth'))
            torch.save(sh_model.state_dict(), os.path.join(sh_save_path, f'{conf.dataset}_{conf.mode}_shadow_model_forget{conf.forget_class_idx}.pth'))
    
    elif conf.mode == 'unlearn':

        trg_model.load_state_dict(torch.load(f"{trg_save_path}/{conf.dataset}_original_target_model.pth"))
        print("======================Before Unlearning======================")
        if conf.evaluate:
            train_forget_acc, train_forget_loss = evaluate(trg_model, trg_train_forget_loader, loss_fn, conf)
            train_retain_acc, train_forget_loss = evaluate(trg_model, trg_train_retain_loader, loss_fn, conf)
            test_forget_acc, test_forget_loss = evaluate(trg_model, trg_test_forget_loader, loss_fn, conf)
            test_retain_acc, train_retain_loss = evaluate(trg_model, trg_test_retain_loader, loss_fn, conf)
            print(f'Target Train Forget Acc: {train_forget_acc*100:.4f}|Target Train Retain Acc: {train_retain_acc*100:.4f}|Target Test Forget Acc: {test_forget_acc*100:.4f}|Target Test Retain Acc: {test_retain_acc*100:.4f}\n')

        print("Start Unlearning...\n")
        training_time = time()
        optimizer = optim.SGD(trg_model.parameters(), lr=conf.unlearn_lr, momentum=conf.unlearn_momentum, weight_decay=conf.unlearn_weight_decay)
        unlearn_model = Unlearn(trg_model, trg_train_forget_loader, conf)
        torch.cuda.synchronize()
        print(f'Unlearning time: {time()-training_time:.3f}\n')
        print("Finish Unlearning...\n")

        print("======================After Unlearning======================")
        if conf.evaluate:
            train_forget_acc, train_forget_loss = evaluate(unlearn_model, trg_train_forget_loader, loss_fn, conf)
            train_retain_acc, train_forget_loss = evaluate(unlearn_model, trg_train_retain_loader, loss_fn, conf)
            test_forget_acc, test_forget_loss = evaluate(unlearn_model, trg_test_forget_loader, loss_fn, conf)
            test_retain_acc, train_retain_loss = evaluate(unlearn_model, trg_test_retain_loader, loss_fn, conf)
            print(f'Train Forget Acc: {train_forget_acc*100:.4f}|Train Retain Acc: {train_retain_acc*100:.4f}|Test Forget Acc: {test_forget_acc*100:.4f}|Test Retain Acc: {test_retain_acc*100:.4f}\n')
            with open(os.path.join(conf.save_dir, f'{conf.dataset}_{conf.mode}_result_forget{conf.forget_class_idx}.txt'), 'w') as f:
                f.write(f'forget_class_idx: {conf.forget_class_idx}|unlearn_epoch: {conf.unlearn_epochs}|unlearn_lr: {conf.unlearn_lr}\n')
                f.write(f'Train Forget Acc: {train_forget_acc*100:.4f}|Train Retain Acc: {train_retain_acc*100:.4f}|Test Forget Acc: {test_forget_acc*100:.4f}|Test Retain Acc: {test_retain_acc*100:.4f}\n')
        print('Saving model...')
        torch.save(unlearn_model.state_dict(), os.path.join(trg_save_path, f'{conf.dataset}_{conf.mode}_model_forget{conf.forget_class_idx}.pth'))
