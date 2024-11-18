from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision

from dataclasses import dataclass
from pytorch_kid34k.data_utils.dataset import get_dataset as get_real_dataset
import pickle
import torch.nn.functional as F
from pytorch_lightning.metrics import Accuracy

@dataclass
class MIADataloader:
    target_trainloader: DataLoader
    target_valloader: DataLoader
    shadow_trainloader: DataLoader
    shadow_valloader: DataLoader


# for mia
def get_dataset(conf, train=False, args=None, save_folder='', small_test = False, unlearn=False):
    if conf.data_type == "cifar10":
        if unlearn:
            transform=transforms.Compose([
            transforms.ToTensor(),
            ])
            dataset = torchvision.datasets.CIFAR10(
                root=conf.data_dir, train=train, download=True, transform=transform)
        else:
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            dataset = torchvision.datasets.CIFAR10(
                root=conf.data_dir, train=train, download=True, transform=transform)


        return dataset

    elif conf.data_type == "cifar100":
        if unlearn:
            transform=transforms.Compose([
            transforms.ToTensor(),
            ])
            dataset = torchvision.datasets.CIFAR100(
                root=conf.data_dir, train=train, download=True, transform=transform)
        else:
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            dataset = torchvision.datasets.CIFAR100(root=conf.data_dir, train=train, download=True, transform=transform)
        return dataset

    elif conf.data_type == "kid34k":
        if args.ae_test:
            S_trainset, S_validset, S_testset = get_real_dataset(args.s_train_data_json_path, args.s_test_data_json_path, args.print_test, args.screen_test, small_test)
            V_trainset, V_validset, V_testset = get_real_dataset(args.v_org_train_data_json_path, args.v_org_test_data_json_path, args.print_test, args.screen_test, small_test)
            print(f"Shadow train size: {len(S_trainset)}, Shadow valid size: {len(S_validset)}, Shadow test size: {len(S_testset)}")
            print(f"Victim train size: {len(V_trainset)}, Victim valid size: {len(V_validset)}, Victim test size: {len(V_testset)}")
        else:
            S_trainset, S_validset, S_testset = get_real_dataset(args.s_org_train_data_json_path, args.s_org_test_data_json_path, args.print_test, args.screen_test, small_test)
            V_trainset, V_validset, V_testset = get_real_dataset(args.v_org_train_data_json_path, args.v_org_test_data_json_path, args.print_test, args.screen_test, small_test)
            print(f"Shadow train size: {len(S_trainset)}, Shadow valid size: {len(S_validset)}, Shadow test size: {len(S_testset)}")
            print(f"Victim train size: {len(V_trainset)}, Victim valid size: {len(V_validset)}, Victim test size: {len(V_testset)}")


        if args.dataset_name == 'cheapfake_ver1':
            data_index_path = f"{save_folder}/data_index.pkl"
            with open(data_index_path, 'rb') as f:
                victim_train_list, victim_test_list, shadow_train_list, shadow_test_list = pickle.load(f)

            V_total_dataset = ConcatDataset([V_trainset, V_validset, V_testset])
            S_total_dataset = ConcatDataset([S_trainset, S_validset, S_testset])

            victim_train_dataset = Subset(V_total_dataset, victim_train_list)
            victim_test_dataset = Subset(V_total_dataset, victim_test_list)
            shadow_train_dataset = Subset(S_total_dataset, shadow_train_list)
            shadow_test_dataset = Subset(S_total_dataset, shadow_test_list)

        else:
            V_total_dataset =  ConcatDataset([V_trainset, V_validset])
            S_total_dataset =  ConcatDataset([S_trainset, S_validset])

            victim_train_dataset = V_total_dataset
            victim_test_dataset = V_testset
            shadow_train_dataset = S_total_dataset
            shadow_test_dataset = S_testset

        return victim_train_dataset, victim_test_dataset, shadow_train_dataset, shadow_test_dataset


def get_probs(conf, model, dataloader, m_path, mu_test=False):
    # modified from https://github.com/Machine-Learning-Security-Lab/mia_prune
    total_loss = 0.0
    total_acc = 0
    logits, probs, sens, trgs, idx_li = [], [], [], [], []
    if conf.data_type == 'kid34k':
        model.load(m_path)
    else:
        if mu_test:
            model = torch.load(m_path, map_location=f'cuda:{conf.device}')
        else:
            model.load_state_dict(torch.load(m_path, map_location=f'cuda:{conf.device}'))
        model.eval()
        model.to(conf.device)

    CE = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy()
    with torch.no_grad():
        if conf.data_type == 'cifar10':
            for batch in dataloader:
                if len(batch) == 2:
                    images, labels = batch
                elif len(batch) == 3:
                    images, labels, idx = batch

                images, labels = images.to(conf.device), labels.to(conf.device)

                if mu_test:
                    logit = model.forward(images)
                    loss = CE(logit, labels)
                    # pred = logit.argmax(dim=1, keepdim=True)
                    accuracy = accuracy_metric(logit.detach().cpu(), labels.detach().cpu())
                else:
                    loss, accuracy, logit = model.forward((images, labels))

                total_loss += loss.item()
                total_acc += accuracy.item()
                prob = F.softmax(logit, dim=-1)

                if len(images.size()) == 4:
                    x = images.repeat((conf.m, 1, 1, 1))
                elif len(images.size()) == 3:
                    x = images.repeat((conf.m, 1, 1))
                elif len(images.size()) == 2:
                    x = images.repeat((conf.m, 1))

                y = labels.repeat((conf.m))

                u = torch.randn_like(x)
                evaluation_points = x + conf.epsilon * u
                if mu_test:
                    new_logit = model.forward(evaluation_points)
                else:
                    _, _, new_logit = model.forward((evaluation_points, y))
                new_probs = F.softmax(new_logit, dim=-1)
                diff = torch.abs(new_probs - prob.repeat((conf.m, 1)))
                diff = diff.view(conf.m, -1, conf.n_classes)
                sensitivity = diff.mean(dim=0) / conf.epsilon

                sens.append(sensitivity.detach().cpu())
                logits.append(logit.detach().cpu())
                probs.append(prob.detach().cpu())
                trgs.append(labels.detach().cpu())

                if len(batch) == 3:
                    idx_li.append(idx)

        elif conf.data_type == 'cifar100':
            for batch in dataloader:
                if len(batch) == 2:
                    images, labels = batch
                elif len(batch) == 3:
                    images, labels, idx = batch

                labels, images = labels.to(conf.device), images.to(conf.device)
                output = model(images)
                loss = CE(output, labels)
                total_loss += loss.item()
                _, pred = output.max(1)
                total_acc += pred.eq(labels).sum()
                prob = F.softmax(output, dim=-1)

                if len(images.size()) == 4:
                    x = images.repeat((conf.m, 1, 1, 1))
                elif len(images.size()) == 3:
                    x = images.repeat((conf.m, 1, 1))
                elif len(images.size()) == 2:
                    x = images.repeat((conf.m, 1))

                u = torch.randn_like(x)
                evaluation_points = x + conf.epsilon * u
                new_probs = F.softmax(model(evaluation_points), dim=-1)
                diff = torch.abs(new_probs - prob.repeat((conf.m, 1)))
                diff = diff.view(conf.m, -1, conf.n_classes)
                sensitivity = diff.mean(dim=0) / conf.epsilon

                sens.append(sensitivity.detach().cpu())
                logits.append(output.detach().cpu())
                probs.append(prob.detach().cpu())
                trgs.append(labels.detach().cpu())

                if len(batch) == 3:
                    idx_li.append(idx)

        elif conf.data_type == 'kid34k':
            avg_loss, avg_acc, logits, probs, sens, trgs = model.get_logit(dataloader, conf.m, conf.epsilon, conf.n_classes)

        elif conf.data_type == 'kr_celeb':
            total = 0
            for batch in dataloader:
                if len(batch) == 2:
                    images, labels = batch
                elif len(batch) == 3:
                    images, labels, idx = batch

                images, labels = images.to(conf.device), labels.to(conf.device)
                output = model(images)
                loss=CE(output, labels)

                pred = output.argmax(dim=1, keepdim=True)
                total_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_loss += loss.item() * len(labels)
                total += labels.size(0)

                prob = F.softmax(output, dim=-1)

                if len(images.size()) == 4:
                    x = images.repeat((conf.m, 1, 1, 1))
                elif len(images.size()) == 3:
                    x = images.repeat((conf.m, 1, 1))
                elif len(images.size()) == 2:
                    x = images.repeat((conf.m, 1))

                u = torch.randn_like(x)
                evaluation_points = x + conf.epsilon * u
                new_probs = F.softmax(model(evaluation_points), dim=-1)
                diff = torch.abs(new_probs - prob.repeat((conf.m, 1)))
                diff = diff.view(conf.m, -1, conf.n_classes)
                sensitivity = diff.mean(dim=0) / conf.epsilon

                sens.append(sensitivity.detach().cpu())
                logits.append(output.detach().cpu())
                probs.append(prob.detach().cpu())
                trgs.append(labels.detach().cpu())

                if len(batch) == 3:
                    idx_li.append(idx)

    logits = torch.cat(logits, dim=0) if logits else None
    probs = torch.cat(probs, dim=0) if probs else None
    trgs = torch.cat(trgs, dim=0) if len(trgs) > 0 else None
    sens = torch.cat(sens, dim=0) if sens else None
    idx_li = torch.cat(idx_li, dim=0) if idx_li else None

    if conf.data_type == 'cifar10':
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
    elif conf.data_type == 'cifar100':
        avg_loss = total_loss / len(dataloader.dataset)
        avg_acc = total_acc / len(dataloader.dataset)
    elif conf.data_type == 'kr_celeb':
        avg_loss = total_loss / total
        avg_acc = total_acc / total

    if idx_li is None:
        return avg_loss, avg_acc, logits, probs, sens, trgs
    else:
        return avg_loss, avg_acc, logits, probs, sens, trgs, idx_li



class SiameseAttackDataset(Dataset):
    def __init__(self, train_probs, test_probs, trg_tr_idx_li, trg_tt_idx_li, is_mem) -> None:
        # super().__init__()
        self.is_mem = is_mem

        if (trg_tr_idx_li is not None) and (trg_tt_idx_li is not None):
            trg_tr_idx_li = trg_tr_idx_li.unsqueeze(1)
            trg_tt_idx_li = trg_tt_idx_li.unsqueeze(1)

            train_probs = torch.cat([train_probs, trg_tr_idx_li], dim=1)
            test_probs = torch.cat([test_probs, trg_tt_idx_li], dim=1)


        min_len = min(len(train_probs), len(test_probs))
        train_probs = train_probs[:min_len]
        test_probs = test_probs[:min_len]

        shuff_idx = torch.randperm(min_len)
        self.train_probs = train_probs[shuff_idx]
        self.test_probs = test_probs[shuff_idx]

        div_tr_len = len(self.train_probs) // 2
        self.train_probs1 = self.train_probs[:div_tr_len]
        self.train_probs2 = self.train_probs[div_tr_len:]

        div_tt_len = len(self.test_probs) // 2
        self.test_probs1 = self.test_probs[:div_tt_len]
        self.test_probs2 = self.test_probs[div_tt_len:]

        self.data = []
        self.init_label()

    def __len__(self):
        return len(self.data)

    def build_pairs(self, v1, v2, label):
        for i, j in zip(v1, v2):
            i = i.cpu().detach()
            j = j.cpu().detach()
            self.data.append((i, j, label))

    def init_label(self):
        if self.is_mem is not None:
            # Attkacker has member samples
            if self.is_mem == True:
                self.build_pairs(self.train_probs1, self.test_probs1, 0)
                self.build_pairs(self.train_probs1, self.train_probs2, 1)
            # Attkacker has non-member samples
            else:
                self.build_pairs(self.test_probs1, self.train_probs1, 0)
                self.build_pairs(self.test_probs1, self.test_probs2, 1)
        else:
            self.build_pairs(self.train_probs1, self.test_probs1, 0)
            self.build_pairs(self.train_probs2, self.test_probs2, 0)
            self.build_pairs(self.train_probs1, self.train_probs2, 1)
            self.build_pairs(self.test_probs1, self.test_probs2, 1)

    def __getitem__(self, idx):
        return self.data[idx]

def get_siam_attack_dataset(train_probs, test_probs, trg_tr_idx_li=None, trg_tt_idx_li=None, is_mem = None):
    attack_dataset = SiameseAttackDataset(train_probs, test_probs, trg_tr_idx_li, trg_tt_idx_li, is_mem)

    return attack_dataset