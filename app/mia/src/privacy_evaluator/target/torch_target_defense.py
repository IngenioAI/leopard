'''
modified from
https://github.com/DingfanChen/RelaxLoss/

'''

import torchvision.transforms as transforms
# from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset
import pickle
import random
import shutil
# from attacks.config import aconf, device
# from _utils.data import TData

import argparse
from functools import partial

# dp module
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from defense_utils import *
from resnet import resnet20, resnet50
from densnet import densenet121
from vgg import vgg11

'''
{
    "dataset_name": "cifar10",
    "model_name": "resnet18", "resnet50", 'densenet121'
    "epochs": 100,
    "num_cls": 10,
    "input_dim": 3,
    "optimizer": "adam",
    "lr": 0.001,
    "weight_decay": 5e-4
    "alpha": 1
    "upper": 1
}

{
    "dataset_name": "cifar100",
    "model_name": "resnet18", "resnet50", 'densenet121'
    "epochs": 100,
    "num_cls": 100,
    "input_dim": 3,
    "optimizer": "adam",
    "lr": 0.001,
    "weight_decay": 5e-4
    "alpha": 3
    "upper" : 1
}
'''


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            nn.init.xavier_normal_(m.weight)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()


def ds_to_numpy(trainset, testset):
    x_train = trainset.data
    x_test = testset.data
    y_train = trainset.targets
    y_test = testset.targets
    x = np.concatenate([x_train, x_test]).astype(np.float32) / 255
    y = np.concatenate([y_train, y_test]).astype(np.int32).squeeze()

    return x, y


def group_data(data, label):
    gr_data = []
    for i, j in zip(data, label):
        gr_data.append([i, j])
    return gr_data


def load_torch_cifar10(train=True):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='/dataset/cifar10-data', train=train, download=True, transform=transform)

    return dataset


def load_torch_cifar100(train=True):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = torchvision.datasets.CIFAR100(
        root='/dataset/cifar100-data', train=train, download=True, transform=transform)

    return dataset


'''
attack models arch...
'''


class ColumnFC(nn.Module):
    def __init__(self, input_dim=100, output_dim=100, dropout=0.1):
        super(ColumnFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


'''
defense method objective function
'''


# label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    '''
    epsilon * 1/K \sum_{i=1}^K -\log p(yi|x) + (1-epsilon)* \sum_{i=1}^K -ti \log p(yi|x)
    =  epsilon * KL (u || p(y|x)) + const + (1-epsilon)* CE(p(y|x), target)
    '''

    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.linear_combination(loss / n, nll, self.epsilon)

    @staticmethod
    def linear_combination(x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


# confidence masking
class ConfidenceMasking(nn.Module):
    def __init__(self, criterion, alpha: float = 0.1, reduction='mean'):
        super().__init__()
        self.criterion = criterion
        self.alpha = alpha
        self.reduction = reduction
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, target):
        loss = self.criterion(preds, target)
        probs = self.softmax(preds)
        logprobs = self.logsoftmax(preds)
        entropy = self.reduce_loss(torch.mul(probs, logprobs).sum(dim=-1), self.reduction)  # = negated entropy
        return loss + self.alpha * entropy

    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


'''
target model arch
'''


class ResNet:
    def __init__(self, device, save_folder, num_cls, epochs, model_name,
                 lr, weight_decay, momentum, input_dim, dropout):

        self.num_cls = num_cls
        self.save_pref = save_folder

        if model_name == 'resnet20':
            self.model = resnet20(num_classes=self.num_cls, droprate=dropout)

        elif model_name == 'resnet50':
            self.model = resnet50(num_classes=self.num_cls, droprate=dropout)

        elif model_name == 'densenet121':
            self.model = densenet121(droprate=dropout, num_class=self.num_cls, pretrained=False)

        elif model_name == 'vgg11':
            self.model = vgg11(num_classes=self.num_cls, droprate=dropout)

        self.device = device
        self.model.to(self.device)
        self.model.apply(weight_init)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

        # if epochs == 0:
        #     self.scheduler = None
        # else:
        #     self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #         self.optimizer, milestones=[epochs // 2, epochs * 3 // 4], gamma=0.1)

        '''
        attack models..
        '''
        # self.attack_model = ColumnFC(input_dim=num_cls*2, output_dim=2)
        # self.attack_model.to(device)
        # self.attack_model.apply(weight_init)
        # self.attack_model_optim = torch.optim.SGD(self.attack_model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_ls = LabelSmoothingCrossEntropy(epsilon=0.1)  # epsilon for label smoothing
        self.criterion_cm = ConfidenceMasking(criterion=nn.CrossEntropyLoss(),
                                              alpha=0.1)  # alpha : weight for the entropy term

        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.crossentropy_soft = partial(CrossEntropy_soft, reduction='none')
        self.crossentropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    # none, dropout
    def train(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            adjust_learning_rate(self.optimizer, epoch, 0.1, [150, 225])
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        # if self.scheduler:
        #     self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        return acc, total_loss

    # label-smoothing
    def train_label_smoothing(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            adjust_learning_rate(self.optimizer, epoch, 0.1, [150, 225])
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion_ls(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        # if self.scheduler:
        #     self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        return acc, total_loss

    # confidence-masking
    def train_confidence_masking(self, train_loader, epoch, log_pref=""):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            adjust_learning_rate(self.optimizer, epoch, 0.1, [150, 225])
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion_cm(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        # if self.scheduler:
        #     self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    # relaxloss
    def train_relaxloss(self, train_loader, epoch, num_class, upper, alpha, log_pref=""):
        '''
        cifar10 : alpha(1), upper(1) / cifar100 : alpha(3), upper(1)
        alpha = 1 # target loss leve
        upper = 1 # maximum confidence level
        '''

        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            adjust_learning_rate(self.optimizer, epoch, 0.1, [150, 225])
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss_ce_full = self.crossentropy_noreduce(outputs, targets)
            loss_ce = torch.mean(loss_ce_full)

            if epoch % 2 == 0:
                loss = (loss_ce - alpha).abs()
            else:
                if loss_ce > alpha:  # normal gradient descent
                    loss = loss_ce
                else:  # posterior flattening
                    pred = torch.argmax(outputs, dim=1)
                    correct = torch.eq(pred, targets).float()
                    confidence_target = self.softmax(outputs)[torch.arange(targets.size(0)), targets]
                    confidence_target = torch.clamp(confidence_target, min=0., max=upper)
                    confidence_else = (1.0 - confidence_target) / (num_class - 1)
                    onehot = one_hot_embedding(targets, num_class)
                    soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, num_class) \
                                   + (1 - onehot) * confidence_else.unsqueeze(-1).repeat(1, num_class)
                    loss = (1 - correct) * self.crossentropy_soft(outputs, soft_targets) - 1. * loss_ce_full
                    loss = torch.mean(loss)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        # if self.scheduler:
        #     self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        # if log_pref:
        #     print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    def test(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                if isinstance(self.criterion, nn.BCELoss):
                    correct += torch.sum(torch.round(outputs) == targets)
                else:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        acc = 100. * correct / total
        total_loss /= total
        return acc, total_loss

    def save(self, epoch):
        save_path = f"{self.save_pref}/{epoch}.pt"
        torch.save(self.model, save_path)
        return save_path

    def load(self, load_path, verbose=False):
        state = torch.load(load_path, map_location=self.device)
        acc = state['acc']
        if verbose:
            print(f"Load model from {load_path}")
            print(f"Epoch {state['epoch']}, Acc: {state['acc']:.3f}, Loss: {state['loss']:.3f}")
        self.model.load_state_dict(state['state'])
        return acc

    # for self-attention attack
    def predict_target_sensitivity(self, data_loader, m=10, epsilon=1e-3):
        self.model.eval()
        predict_list = []
        sensitivity_list = []
        target_list = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predicts = F.softmax(outputs, dim=-1)
                predict_list.append(predicts.detach().data.cpu())
                target_list.append(targets)

                if len(inputs.size()) == 4:
                    x = inputs.repeat((m, 1, 1, 1))
                elif len(inputs.size()) == 3:
                    x = inputs.repeat((m, 1, 1))
                elif len(inputs.size()) == 2:
                    x = inputs.repeat((m, 1))
                u = torch.randn_like(x)
                evaluation_points = x + epsilon * u
                new_predicts = F.softmax(self.model(evaluation_points), dim=-1)
                diff = torch.abs(new_predicts - predicts.repeat((m, 1)))
                diff = diff.view(m, -1, self.num_cls)
                sensitivity = diff.mean(dim=0) / epsilon
                sensitivity_list.append(sensitivity.detach().data.cpu())

        targets = torch.cat(target_list, dim=0)
        predicts = torch.cat(predict_list, dim=0)
        sensitivities = torch.cat(sensitivity_list, dim=0)
        return predicts, targets, sensitivities


class DPsgd:
    def __init__(self, device, save_folder, num_cls, epochs, model_name,
                 lr, weight_decay, momentum, input_dim, dropout):

        self.num_cls = num_cls
        self.save_pref = save_folder

        if model_name == 'resnet20':
            self.model = resnet20(num_classes=self.num_cls, droprate=dropout)

        elif model_name == 'resnet50':
            self.model = resnet50(num_classes=self.num_cls, droprate=dropout)

        elif model_name == 'densenet121':
            self.model = densenet121(droprate=dropout, num_class=self.num_cls, pretrained=False)

        elif model_name == 'vgg11':
            self.model = vgg11(num_classes=self.num_cls, droprate=dropout)

        self.model = ModuleValidator.fix(self.model)
        ModuleValidator.validate(self.model, strict=False)

        self.model.to(device)
        self.model.apply(weight_init)
        self.device = device

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        if epochs == 0:
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[epochs // 2, epochs * 3 // 4], gamma=0.1)

        self.criterion = nn.CrossEntropyLoss()

    # dpsgd
    def train_dpsgd(self, train_loader, epoch, batch_size, log_pref=""):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # noise_multiplier : noise multiplier (small value for better utility)
        # max_grad_norm : grad norm clipping bound
        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            noise_multiplier=0.01,
            max_grad_norm=1.2,
        )

        for inputs, targets in train_loader:
            adjust_learning_rate(self.optimizer, epoch, 0.1, [150, 225])
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        # if self.scheduler:
        #     self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    def test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                if isinstance(self.criterion, nn.BCELoss):
                    correct += torch.sum(torch.round(outputs) == targets)
                else:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    def save(self, epoch):
        save_path = f"{self.save_pref}/{epoch}.pt"
        state = {
            'epoch': epoch + 1,
            'state': self.model
        }
        torch.save(self.model, save_path)
        return save_path

    def load(self, load_path, verbose=False):
        state = torch.load(load_path, map_location=self.device)
        acc = state['acc']
        if verbose:
            print(f"Load model from {load_path}")
            print(f"Epoch {state['epoch']}, Acc: {state['acc']:.3f}, Loss: {state['loss']:.3f}")
        self.model.load_state_dict(state['state'])
        return acc

    # for self-attention attack
    def predict_target_sensitivity(self, data_loader, m=10, epsilon=1e-3):
        self.model.eval()
        predict_list = []
        sensitivity_list = []
        target_list = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predicts = F.softmax(outputs, dim=-1)
                predict_list.append(predicts.detach().data.cpu())
                target_list.append(targets)

                if len(inputs.size()) == 4:
                    x = inputs.repeat((m, 1, 1, 1))
                elif len(inputs.size()) == 3:
                    x = inputs.repeat((m, 1, 1))
                elif len(inputs.size()) == 2:
                    x = inputs.repeat((m, 1))
                u = torch.randn_like(x)
                evaluation_points = x + epsilon * u
                new_predicts = F.softmax(self.model(evaluation_points), dim=-1)
                diff = torch.abs(new_predicts - predicts.repeat((m, 1)))
                diff = diff.view(m, -1, self.num_cls)
                sensitivity = diff.mean(dim=0) / epsilon
                sensitivity_list.append(sensitivity.detach().data.cpu())

        targets = torch.cat(target_list, dim=0)
        predicts = torch.cat(predict_list, dim=0)
        sensitivities = torch.cat(sensitivity_list, dim=0)
        return predicts, targets, sensitivities


def run(args, epoch_callback=None):
    device = f"cuda:0"
    cudnn.benchmark = True
    save_dir = f"/model/{args.datasets}_{args.model_name}/{args.defense}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"save dir: {save_dir}")

    if args.datasets == 'cifar10':
        trainset = load_torch_cifar10(train=True)
        testset = load_torch_cifar10(train=False)
        num_cls = 10
        upper = 1
        alpha = 1
    elif args.datasets == 'cifar100':
        trainset = load_torch_cifar100(train=True)
        testset = load_torch_cifar100(train=False)
        upper = 1
        alpha = 3
        num_cls = 100

    if testset is None:
        total_dataset = trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)

    target_li, shadow_li = train_test_split(list(range(total_size)), test_size=0.5, random_state=args.seed)
    target_train_li, target_test_li = train_test_split(target_li, test_size=0.45, random_state=args.seed)
    target_train_li, target_valid_li = train_test_split(
        target_train_li, test_size=0.1818, random_state=args.seed)

    for i in range(args.shadow_num):
        shadow_train_li, shadow_test_li = train_test_split(
            shadow_li, test_size=0.45, random_state=args.seed + i)
        shadow_train_li, shadow_valid_li = train_test_split(
            shadow_train_li, test_size=0.1818, random_state=args.seed + i)

    # # shadow/target dataset info save
    # with open(f"{data_dir}/shadow_dataset_index.pkl", 'wb') as f:
    #    pickle.dump([shadow_train_li, shadow_valid_li, shadow_test_li], f)

    # with open(f"{data_dir}/target_dataset_index.pkl", 'wb') as f:
    #    pickle.dump([target_train_li, target_valid_li, target_test_li], f)

    target_train_dataset = Subset(total_dataset, target_train_li)
    target_valid_dataset = Subset(total_dataset, target_valid_li)
    target_test_dataset = Subset(total_dataset, target_test_li)

    print(f"total data size: {total_size}, "
          f"target train size: {len(target_train_li)}, "
          f"target valid size: {len(target_valid_li)}, "
          f"target test size: {len(target_test_li)}")

    target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True, worker_init_fn=seed_worker)
    target_valid_loader = DataLoader(target_valid_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=True, worker_init_fn=seed_worker)
    target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True, worker_init_fn=seed_worker)

    ### train target model ###
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    if args.defense in ['none', 'es', 'ls', 'cm', 'dropout', 'relaxloss']:
        if args.defense == 'dropout':
            model = ResNet(device, save_dir, num_cls, args.epochs, args.model_name, lr=args.lr,
                           weight_decay=args.weight_decay, momentum=args.momentum, input_dim=100,
                           dropout=0.5)  # dropout rate setting
        else:
            model = ResNet(device, save_dir, num_cls, args.epochs, args.model_name, lr=args.lr,
                           weight_decay=args.weight_decay, momentum=args.momentum, input_dim=100, dropout=0)

        best_acc = 0
        count = 0
        for epoch in range(args.epochs):
            if args.defense in ['none', 'es', 'dropout']:
                train_acc, train_loss = model.train(target_train_loader, epoch)
            elif args.defense == 'ls':  # label_smoothing
                train_acc, train_loss = model.train_label_smoothing(target_train_loader, epoch)
            elif args.defense == 'cm':  # confidence_masking
                train_acc, train_loss = model.train_confidence_masking(target_train_loader, epoch)
            elif args.defense == 'relaxloss':
                train_acc, train_loss = model.train_relaxloss(target_train_loader, epoch, num_cls, upper, alpha)
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            print(f"Train epoch {epoch} acc={train_acc}, loss={train_loss}")
            val_acc, val_loss = model.test(target_valid_loader)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = model.save(epoch)
                best_path = save_path
                count = 0
            elif args.defense == 'es' and args.early_stop > 0:
                count += 1
                if count > args.early_stop:
                    print(f"Early Stop at Epoch {epoch}")
                    break
            if epoch_callback is not None:
                epoch_callback(epoch, args.epochs, train_acc_list, train_loss_list, val_acc_list, val_loss_list)

        # shutil.copyfile(best_path, f"{save_dir}/ep_{epoch}_best[{best_acc}].pt")
        shutil.copyfile(best_path, f"{save_dir}/best_model.pt")
    elif args.defense == 'dpsgd':
        model = DPsgd(device, save_dir, num_cls, args.epochs, args.model_name, lr=args.lr,
                      weight_decay=args.weight_decay, momentum=args.momentum, input_dim=100, dropout=0)
        best_acc = 0
        count = 0
        for epoch in range(args.epochs):
            train_acc, train_loss = model.train_dpsgd(target_train_loader, epoch, args.batch_size,
                                                      f"epoch {epoch} train")
            val_acc, val_loss = model.test(target_valid_loader, f"epoch {epoch} valid")
            # test_acc, test_loss = model.test(target_test_loader, f"epoch {epoch} test")
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = model.save(epoch)
                best_path = save_path
                count = 0
            # elif args.early_stop > 0:
            #     count += 1
            #     if count > args.early_stop:
            #         print(f"Early Stop at Epoch {epoch}")
            #         break
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            if epoch_callback is not None:
                epoch_callback(epoch, args.epochs, train_acc_list, train_loss_list, val_acc_list, val_loss_list)
        shutil.copyfile(best_path, f"{save_dir}/best_model.pt")

    return {
        "defense": args.defense,
        "best_path": best_path,
        "best_acc": best_acc,
        "train_acc_list": train_acc_list,
        "train_loss_list": train_loss_list,
        "val_acc_list": val_acc_list,
        "val_loss_list": val_loss_list
    }


if __name__ == '__main__':
    ''''
    epochs = 50
    seed=42
    shadow_num=1
    batch_size = 64
    early_stop = 5
    num_workers = 10
    '''
    parser = argparse.ArgumentParser(description='MIA - Target Training')

    parser.add_argument('-m', '--model_name', type=str,
                        help='please input dataset : resnet20, resnet50, densenet121, vgg11')
    parser.add_argument('-d', '--datasets', type=str,
                        help='please input dataset : cifar10, cifar100')
    parser.add_argument('-df', '--defense', type=str,
                        help='please input defense method : none, ls, cm, dropout, relaxloss, dpsgd, es')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of processes')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--shadow_num', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1000)

    args = parser.parse_args()

    run(args)
