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
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import defaultdict
from facenet_pytorch import InceptionResnetV1

def print_args(args):
    print("\n#### configurations ####")
    for k, v in vars(args).items():
            print('{}: {}'.format(k, v))
    print("########################\n")

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_dataset(args, trans=None):
    if trans is None:
        trans = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
            ])

    trainset = ImageFolder(os.path.join(args.data_path, 'train'), transform=trans)
    testset = ImageFolder(os.path.join(args.data_path, 'test'), transform=trans)
    return trainset, testset

def get_dataloader(train_dataset, test_dataset, args):
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def split_class_data(dataset, forget_class_idx):
    targets = torch.tensor(dataset.targets)
    forget_indices = (targets == forget_class_idx).nonzero(as_tuple=True)[0].tolist()
    remain_indices = (targets != forget_class_idx).nonzero(as_tuple=True)[0].tolist()
    return forget_indices, remain_indices

def get_unlearn_loader(train_set, test_set, args):
    indices_path = os.path.join(args.data_path, 'dataset_indices')
    train_indices_path = os.path.join(indices_path, f'train_indices_{args.forget_class_idx}.pt')
    test_indices_path = os.path.join(indices_path, f'test_indices_{args.forget_class_idx}.pt')
    if os.path.exists(train_indices_path) and os.path.exists(test_indices_path):
        print(f"Load indices from {train_indices_path} and {test_indices_path}")
        train_indices = torch.load(train_indices_path)
        test_indices = torch.load(test_indices_path)
        train_forget_indices = train_indices['forget']
        train_remain_indices = train_indices['remain']
        test_forget_indices = test_indices['forget']
        test_remain_indices = test_indices['remain']
    else:
        train_forget_indices, train_remain_indices = split_class_data(train_set, args.forget_class_idx)
        test_forget_indices, test_remain_indices = split_class_data(test_set, args.forget_class_idx)
        train_indices = {'forget': train_forget_indices, 'remain': train_remain_indices}
        test_indices = {'forget': test_forget_indices, 'remain': test_remain_indices}
        os.makedirs(indices_path, exist_ok=True)
        torch.save(train_indices, train_indices_path)
        torch.save(test_indices, test_indices_path)

    train_forget_set = Subset(train_set, train_forget_indices)
    train_retain_set = Subset(train_set, train_remain_indices)

    test_forget_set = Subset(test_set, test_forget_indices)
    test_retain_set = Subset(test_set, test_remain_indices)

    train_forget_loader = DataLoader(dataset=train_forget_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_retain_loader = DataLoader(dataset=train_retain_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_forget_loader = DataLoader(dataset=test_forget_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_retain_loader = DataLoader(dataset=test_retain_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_forget_loader, train_retain_loader, test_forget_loader, test_retain_loader

def train(model, dataloader, criterion, optimizer, args):
    model.train()
    correct, losses, total=0, 0, 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        images, labels = images.to(args.device), labels.to(args.device)
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
def evaluate(model, dataloader, criterion, args):
    model.eval()
    correct, losses, total = 0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(args.device), labels.to(args.device)
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

def Unlearn(original_model, train_forget_loader, args):
    original_model = original_model.to(args.device)
    unlearn_model = deepcopy(original_model).to(args.device)
    attack = PGD(model=original_model.to(args.device), eps=args.eps, alpha=args.alpha, iters=args.iters, denorm=args.denorm)
    attack.set_normalization(args.mean, args.std)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=args.unlearn_lr, momentum=args.unlearn_momentum, weight_decay=args.unlearn_weight_decay)

    for epoch in range(args.unlearn_epochs):
        original_model.train()
        start = time()
        nearest_label = []
        correct, nums_filpped, total, losses = 0, 0, 0, 0
        unlearn_model.train()
        for images, labels in train_forget_loader:
            images = images.to(args.device)
            _, adv_labels = attack(images, labels, return_adv_labels=True)

            nearest_label.append(adv_labels.tolist())
            nums_filpped += (labels != adv_labels).sum().item()

            outputs = unlearn_model(images)
            loss = criterion(outputs, adv_labels.to(args.device))

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

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default='original', choices=['original', 'retrain', 'unlearn'])
    #### dataset ####
    parser.add_argument("--dataset", type=str, default='VGGFace2')
    parser.add_argument("--data_path", type=str, default='/dataset/dataset-demo')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--mean", type=list, default=[0.5, 0.5, 0.5])
    parser.add_argument("--std", type=list, default=[0.5, 0.5, 0.5])
    #### train & test ####
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    #### save & load ####
    parser.add_argument("--evaluate", default=True)
    parser.add_argument("--save_root_dir", type=str, default='./')
    #### for unlearn ####
    parser.add_argument("--model_load_path", default='./model-demo.pth')
    parser.add_argument("--forget_class_idx", type=int, default=3)
    parser.add_argument("--eps", type=float, default=32)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--denorm", default=True)
    parser.add_argument("--unlearn_epochs", type=int, default=5)
    parser.add_argument("--unlearn_lr", type=float, default=6e-4)
    parser.add_argument("--unlearn_momentum", type=float, default=0.9)
    parser.add_argument("--unlearn_weight_decay", type=float, default=1e-4)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

def main(args):
    print(f'Setting {args.dataset} dataset...\n')
    train_set, test_set = get_dataset(args)
    train_loader, test_loader = get_dataloader(train_set, test_set, args)
    train_forget_loader, train_retain_loader, test_forget_loader, test_retain_loader = get_unlearn_loader(train_set, test_set, args)

    def get_class_counts(dataset):
        class_counts = defaultdict(int)
        for _, label in dataset.samples:
            class_counts[dataset.classes[label]] += 1
        return class_counts

    train_counts = get_class_counts(train_set)
    test_counts = get_class_counts(test_set)
    for class_name in train_set.classes:
        print(f"{class_name}: Train={train_counts.get(class_name, 0)}, Test={test_counts.get(class_name, 0)}")

    print(f'\nBuilding Model...\n')
    model = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(train_set.class_to_idx)
    ).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss()

    if args.mode in ['original', 'retrain']:
        total_start = time()
        print(f'Start training {args.mode.capitalize()} model...')
        if args.mode == 'retrain':
            train_loader = train_retain_loader
            test_loader = test_retain_loader

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for i in tqdm(range(args.epochs), desc='Training'):
            train_acc, train_loss = train(model, train_loader, loss_fn, optimizer, args)
            print(f'EPOCH {i}/{args.epochs} : Train loss: {train_loss:.4f}')
        torch.cuda.synchronize()

        print(f'Training time: {time()-total_start:.3f}\n')

        test_acc, test_loss = evaluate(model, test_loader, loss_fn, args)
        print(f'Test Acc: {test_acc*100:.4f}|Test Loss: {test_loss:.4f}')

        if args.evaluate:
            train_forget_acc, train_forget_loss = evaluate(model, train_forget_loader, loss_fn, args)
            train_retain_acc, train_forget_loss = evaluate(model, train_retain_loader, loss_fn, args)
            test_forget_acc, test_forget_loss = evaluate(model, test_forget_loader, loss_fn, args)
            test_retain_acc, train_retain_loss = evaluate(model, test_retain_loader, loss_fn, args)
            print(f'Train Forget Acc: {train_forget_acc*100:.4f}|Train Retain Acc: {train_retain_acc*100:.4f}|Test Forget Acc: {test_forget_acc*100:.4f}|Test Retain Acc: {test_retain_acc*100:.4f}\n')
            with open(os.path.join(args.save_root_dir, f'{args.dataset}_{args.mode}_result_forget{args.forget_class_idx}.txt'), 'w') as f:
                f.write(f'forget_class_idx: {args.forget_class_idx}|unlearn_epoch: {args.unlearn_epochs}|unlearn_lr: {args.unlearn_lr}\n')
                f.write(f'Train Forget Acc: {train_forget_acc*100:.4f}|Train Retain Acc: {train_retain_acc*100:.4f}|Test Forget Acc: {test_forget_acc*100:.4f}|Test Retain Acc: {test_retain_acc*100:.4f}\n')
        print('Saving model...\n')
        if args.mode == 'original':
            torch.save(model.state_dict(), os.path.join(args.save_root_dir, f'model-demo.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(args.save_root_dir, f'model-unlearn.pth'))

    elif args.mode == 'unlearn':
        if args.model_load_path is None:
            raise ValueError('Please provide model_load_path for unlearning mode')
        model.load_state_dict(torch.load(args.model_load_path))
        print("======================Before Unlearning======================")
        if args.evaluate:
            train_forget_acc, train_forget_loss = evaluate(model, train_forget_loader, loss_fn, args)
            train_retain_acc, train_forget_loss = evaluate(model, train_retain_loader, loss_fn, args)
            test_forget_acc, test_forget_loss = evaluate(model, test_forget_loader, loss_fn, args)
            test_retain_acc, train_retain_loss = evaluate(model, test_retain_loader, loss_fn, args)
            print(f'Train Forget Acc: {train_forget_acc*100:.4f}|Train Retain Acc: {train_retain_acc*100:.4f}|Test Forget Acc: {test_forget_acc*100:.4f}|Test Retain Acc: {test_retain_acc*100:.4f}\n')

        print("Start Unlearning...\n")
        training_time = time()
        optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=args.unlearn_momentum, weight_decay=args.unlearn_weight_decay)
        unlearn_model = Unlearn(model, train_forget_loader, args)
        torch.cuda.synchronize()
        print(f'Unlearning time: {time()-training_time:.3f}\n')
        print("Finish Unlearning...\n")

        print("======================After Unlearning======================")
        if args.evaluate:
            train_forget_acc, train_forget_loss = evaluate(unlearn_model, train_forget_loader, loss_fn, args)
            train_retain_acc, train_forget_loss = evaluate(unlearn_model, train_retain_loader, loss_fn, args)
            test_forget_acc, test_forget_loss = evaluate(unlearn_model, test_forget_loader, loss_fn, args)
            test_retain_acc, train_retain_loss = evaluate(unlearn_model, test_retain_loader, loss_fn, args)
            print(f'Train Forget Acc: {train_forget_acc*100:.4f}|Train Retain Acc: {train_retain_acc*100:.4f}|Test Forget Acc: {test_forget_acc*100:.4f}|Test Retain Acc: {test_retain_acc*100:.4f}\n')
            with open(os.path.join(args.save_root_dir, f'{args.dataset}_{args.mode}_result_forget{args.forget_class_idx}.txt'), 'w') as f:
                f.write(f'forget_class_idx: {args.forget_class_idx}|unlearn_epoch: {args.unlearn_epochs}|unlearn_lr: {args.unlearn_lr}\n')
                f.write(f'Train Forget Acc: {train_forget_acc*100:.4f}|Train Retain Acc: {train_retain_acc*100:.4f}|Test Forget Acc: {test_forget_acc*100:.4f}|Test Retain Acc: {test_retain_acc*100:.4f}\n')
        print('Saving model...')
        torch.save(unlearn_model.state_dict(), os.path.join(args.save_root_dir, f'{args.dataset}_{args.mode}_model_forget{args.forget_class_idx}.pth'))

if __name__ == '__main__':
    args = build_args()
    print_args(args)
    fix_seed(args.seed)
    main(args)
