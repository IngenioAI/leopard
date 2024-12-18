'''
    FaceNet train/unlearning script
'''
import os
import argparse
import json
import torch
import time

import numpy as np
import random

from copy import deepcopy

from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

def clear_progress():
    if os.path.exists("/apprun/progress.json"):
        os.remove("/apprun/progress.json")

def save_progress(obj):
    with open("/apprun/progress.json", "wt", encoding="UTF-8") as fp:
        json.dump(obj, fp, indent=4)


def save_result(obj, filename):
    with open(f"/data/output/{filename}", "wt", encoding="UTF-8") as fp:
        json.dump(obj, fp, indent=4)

def fix_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def load_data(data_path, batch_size):
    save_progress({
        "status": "running",
        "stage": 1,
        "message": "Data Loading"
    })
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    trainset = ImageFolder(os.path.join(data_path, 'train'), transform=trans)
    testset = ImageFolder(os.path.join(data_path, 'test'), transform=trans)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, trainset.class_to_idx

def split_class_data(dataset, forget_class_idx):
    targets = torch.tensor(dataset.targets)
    forget_indices = (targets == forget_class_idx).nonzero(as_tuple=True)[0].tolist()
    remain_indices = (targets != forget_class_idx).nonzero(as_tuple=True)[0].tolist()
    return forget_indices, remain_indices

def load_unlearning_data(data_path, forget_class_idx, batch_size):
    save_progress({
        "status": "running",
        "stage": 1,
        "message": "Data Loading for Unlearning"
    })
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_set = ImageFolder(os.path.join(data_path, 'train'), transform=trans)
    test_set = ImageFolder(os.path.join(data_path, 'test'), transform=trans)

    indices_path = '/apprun/dataset_indices'
    train_indices_path = os.path.join(indices_path, f'train_indices_{forget_class_idx}.pt')
    test_indices_path = os.path.join(indices_path, f'test_indices_{forget_class_idx}.pt')
    if os.path.exists(train_indices_path) and os.path.exists(test_indices_path):
        print(f"Load indices from {train_indices_path} and {test_indices_path}")
        train_indices = torch.load(train_indices_path)
        test_indices = torch.load(test_indices_path)
        train_forget_indices = train_indices['forget']
        train_remain_indices = train_indices['remain']
        test_forget_indices = test_indices['forget']
        test_remain_indices = test_indices['remain']
    else:
        train_forget_indices, train_remain_indices = split_class_data(train_set, forget_class_idx)
        test_forget_indices, test_remain_indices = split_class_data(test_set, forget_class_idx)
        train_indices = {'forget': train_forget_indices, 'remain': train_remain_indices}
        test_indices = {'forget': test_forget_indices, 'remain': test_remain_indices}
        os.makedirs(indices_path, exist_ok=True)
        torch.save(train_indices, train_indices_path)
        torch.save(test_indices, test_indices_path)

    train_forget_set = Subset(train_set, train_forget_indices)
    #train_retain_set = Subset(train_set, train_remain_indices)

    test_forget_set = Subset(test_set, test_forget_indices)
    test_retain_set = Subset(test_set, test_remain_indices)

    train_forget_loader = DataLoader(dataset=train_forget_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #train_retain_loader = DataLoader(dataset=train_retain_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_forget_loader = DataLoader(dataset=test_forget_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_retain_loader = DataLoader(dataset=test_retain_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_forget_loader, test_forget_loader, test_retain_loader, train_set.class_to_idx

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

class FaceNet():
    def __init__(self, model_path, num_classes=10, resume_train=False, freeze=False):
        self.num_classes = num_classes
        self.freeze = freeze
        self.model_path = model_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = InceptionResnetV1(
            pretrained=None if resume_train else 'vggface2',
            classify=True,
            num_classes=self.num_classes
#           ,dropout_prob=dropout
        )

        if self.freeze:
            # freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

            # unfreeze last layer
            last_layer = list(self.model.modules())[-1]
            print("Freeze all but below layer:")
            print(last_layer)
            print(" ")
            last_layer.weight.requires_grad = True

        self.model.to(self.device)
        if os.path.exists(self.model_path) and resume_train:
            print("Load previous model file:", self.model_path)
            self.model.load_state_dict(torch.load(self.model_path))

    def train_epoch(self, dataloader, criterion, optimizer):
        self.model.train()
        correct, losses, total=0, 0, 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            losses += loss.item() * len(labels)
            total += labels.size(0)
        return correct / total, losses / total

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path

        print('Saving model...', model_path)
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(self.model.state_dict(), model_path)


    def train(self, train_loader, test_loader, epochs=8, lr=1e-3, momentum=0.9, weight_decay=1e-4):
        save_progress({
            "status": "running",
            "stage": 2,
            "message": "Model Training",
        })
        total_start = time.time()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        acc_list = []
        loss_list = []
        for i in range(epochs):
            train_acc, train_loss = self.train_epoch(train_loader, loss_fn, optimizer)
            print(f'EPOCH {i+1}/{epochs}, TrainAcc: {train_acc:.4f}, Train loss: {train_loss:.4f}')
            acc_list.append(train_acc)
            loss_list.append(train_loss)
            save_progress({
                "status": "running",
                "stage": 2,
                "message": f"Training epoch {i+1}/{epochs}",
                "current_epoch": i+1,
                "max_epochs": epochs,
                "acc": acc_list,
                "loss": loss_list
            })

            #scheduler.step()
        torch.cuda.synchronize()
        print(f'Training time: {time.time()-total_start:.3f}\n')
        test_acc, test_loss = self.evaluate(test_loader, loss_fn)
        print(f'Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}')

        save_progress({
            "status": "running",
            "stage": 3,
            "message": "Model Evaluation",
            "acc": test_acc,
            "loss": test_loss
        })
        self.save_model()
        save_progress({
            "status": "done",
            "stage": 4,
            "message": "Model Save",
            "current_epoch": epochs,
            "max_epochs": epochs,
            "acc": acc_list,
            "loss": loss_list,
            "test_acc": test_acc,
            "test_loss": test_loss
        })
        return loss_list, acc_list, test_loss, test_acc

    def unlearn(self, train_forget_loader, test_forget_loader, test_retain_loader, epochs=2, unlearn_model_name=None):
        save_progress({
            "status": "running",
            "stage": 2,
            "message": "Model Unlearning",
        })

        loss_fn = torch.nn.CrossEntropyLoss()

        before_test_forget_acc, _ = self.evaluate(test_forget_loader, loss_fn)
        before_test_retain_acc, _ = self.evaluate(test_retain_loader, loss_fn)

        original_model = self.model.to(self.device)
        unlearn_model = deepcopy(original_model).to(self.device)
        attack = PGD(model=original_model.to(self.device), eps=32, alpha=1, iters=10, denorm=True)
        attack.set_normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=6e-4, momentum=0.9, weight_decay=1e-4)

        loss_list = []
        acc_list = []
        for epoch in range(epochs):
            start = time.time()
            nearest_label = []
            correct, nums_filpped, total, losses = 0, 0, 0, 0
            unlearn_model.train()
            for images, labels in train_forget_loader:
                images = images.to(self.device)
                _, adv_labels = attack(images, labels, return_adv_labels=True)

                nearest_label.append(adv_labels.tolist())
                nums_filpped += (labels != adv_labels).sum().item()

                outputs = unlearn_model(images)
                loss = criterion(outputs, adv_labels.to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = outputs.argmax(dim=1, keepdim=True).detach().cpu()
                correct += pred.eq(labels.view_as(pred)).sum().item()
                total += labels.size(0)
                losses += loss.item() * labels.size(0)
            loss_list.append(losses/total)
            acc_list.append(correct/total)
            torch.cuda.synchronize()
            save_progress({
                "status": "running",
                "stage": 2,
                "message": f"Unlearning epoch {epoch+1}/{epochs}",
                "current_epoch": epoch+1,
                "max_epochs": epochs,
                "loss": loss_list,
                "acc": acc_list
            })
            print(f'Epoch {epoch} | Time {time.time()-start:.3f} | Loss {losses / total:.4f} | Acc {correct/total:.4f} | Flipped {nums_filpped/total:.4f}')

        self.model = unlearn_model
        test_forget_acc, test_forget_loss = self.evaluate(test_forget_loader, loss_fn)
        test_retain_acc, test_retain_loss = self.evaluate(test_retain_loader, loss_fn)

        if unlearn_model_name is not None:
            unlearn_target_path = os.path.join("/model", unlearn_model_name, "model.pth")
            self.save_model(unlearn_target_path)
        else:
            self.save_model()   # overwrite!
        save_progress({
            "status": "done",
            "stage": 4,
            "message": "Model Save",
            "current_epoch": epochs,
            "max_epochs": epochs,
            "loss": loss_list,
            "acc": acc_list
        })
        return test_forget_loss, test_forget_acc, test_retain_loss, test_retain_acc, before_test_forget_acc, before_test_retain_acc

    @torch.no_grad()
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        correct, losses, total = 0, 0, 0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss=criterion(outputs, labels)

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            losses += loss.item() * len(labels)
            total += labels.size(0)

        return correct / total, losses / total

def save_info(info_path, params, metrics):
    with open(info_path, "wt", encoding="utf-8") as fp:
        json.dump({
            "params": params,
            "metrics": metrics
        }, fp)

def save_label(label_path, class_to_idx):
    if label_path is not None:
        label_dir = os.path.dirname(label_path)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        with open(label_path, "wt", encoding="utf-8") as fp:
            json.dump(class_to_idx, fp)

def train(input_params):
    try:
        save_progress({
            "status": "running",
            "stage": 0,
            "message": "Build and download pretrained model..."
        })
        mode = input_params["mode"]
        model_name = input_params["model_name"]
        model_path = os.path.join("/model", model_name, "model.pth")
        data_path = os.path.join("/dataset", input_params["dataset"])
        resume_train = input_params.get("resume_train", False)
        batch_size = input_params.get("batch_size", 128)

        if mode == "train":
            max_epochs = input_params.get("epochs", 8)
            train_loader, test_loader, class_to_idx = load_data(data_path, batch_size)
            num_classes = len(class_to_idx.items())
            model = FaceNet(model_path, num_classes=num_classes, resume_train=resume_train)
            train_loss, train_acc, test_loss, test_acc = model.train(train_loader, test_loader, epochs=max_epochs)

            info_path = os.path.join("/model", model_name, "info.json")
            save_info(info_path, input_params, {"test_acc": test_acc})
            label_path = os.path.join("/model", model_name, "class_to_idx.json")
            save_label(label_path, class_to_idx)

            return {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            }
        elif mode == "unlearn":
            max_epochs = input_params.get("unlearn_epochs", 8)
            forget_class_index = input_params["forget_class_index"]
            train_forget_loader, test_forget_loader, test_retain_loader, class_to_idx = load_unlearning_data(data_path, forget_class_index, batch_size)
            unlearn_model_name = input_params.get("unlearn_model_name", None)
            num_classes = len(class_to_idx.items())
            model = FaceNet(model_path, num_classes=num_classes, resume_train=True)
            test_forget_loss, test_forget_acc, test_retain_loss, test_retain_acc, before_test_forget_acc, before_test_retain_acc = model.unlearn(train_forget_loader, test_forget_loader, test_retain_loader, max_epochs, unlearn_model_name)

            info_path = os.path.join("/model", unlearn_model_name, "info.json")
            save_info(info_path, input_params, {"test_forget_acc": test_forget_acc, "test_retain_acc": test_retain_acc,
                                                "before_test_forget_acc": before_test_forget_acc, "before_test_retain_acc": before_test_retain_acc})
            label_path = os.path.join("/model", unlearn_model_name, "class_to_idx.json")
            save_label(label_path, class_to_idx)

            return {
                "test_forget_loss": test_forget_loss,
                "test_forget_acc": test_forget_acc,
                "test_retain_loss": test_retain_loss,
                "test_retain_acc": test_retain_acc,
                "before_test_forget_acc": before_test_forget_acc,
                "before_test_retain_acc": before_test_retain_acc
            }

    except KeyError as e:
        print("KeyError:", e)
        save_progress({
            "status": "done"
        })
        return [{
            "mode": "train",
            "model_name": "model01",
            "dataset": "dataset01"
        }, {
            "mode": "unlearn",
            "model_name": "model01",
            "dataset": "dataset01",
            "unlearn_model_name": "unlearn01",
            "forget_class_index": 1
        }]

def main(args):
    with open(os.path.join("/data/input", args.input), "rt", encoding="utf-8") as fp:
        input_params = json.load(fp)

    ret = train(input_params)
    save_result(ret, args.output)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()


if __name__ == '__main__':
    clear_progress()
    fix_seed()
    main(parse_arguments())
