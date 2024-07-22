'''
    FaceNet train/unlearning script
'''
import os
import argparse
import json
import torch
import time
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from utils import save_result, clear_progress, save_progress

def load_data(data_path, batch_size):
    save_progress({
        "status": "running",
        "stage": 1,
        "message": f"Data Loading",
    })
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    trainset = ImageFolder(os.path.join(data_path, 'train'), transform=trans)
    testset = ImageFolder(os.path.join(data_path, 'test'), transform=trans)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, trainset.class_to_idx

class FaceNet():
    def __init__(self, model_path, num_classes=10, resume_train=False, freeze=False):
        self.num_classes = num_classes
        self.freeze = freeze
        self.model_path = model_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = InceptionResnetV1(
            pretrained=None if resume_train else 'vggface2',
            classify=True,
            num_classes=self.num_classes,
            dropout_prob=0.2
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

    def train(self, train_loader, test_loader, epochs=8, lr=1e-3, momentum=0.9, weight_decay=1e-4):
        save_progress({
            "status": "running",
            "stage": 2,
            "message": f"Model Training",
        })
        total_start = time.time()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = MultiStepLR(optimizer, [5, 10])
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

            scheduler.step()
        torch.cuda.synchronize()
        print(f'Training time: {time.time()-total_start:.3f}\n')
        test_acc, test_loss = self.evaluate(test_loader, loss_fn)
        print(f'Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}')

        save_progress({
            "status": "running",
            "stage": 3,
            "message": f"Model Evaluation",
            "acc": test_acc,
            "loss": test_loss
        })

        print('Saving model...', self.model_path)
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(self.model.state_dict(), self.model_path)

        save_progress({
            "status": "done",
            "stage": 4,
            "message": f"Model Save"
        })
        return loss_list, acc_list, test_loss, test_acc

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

def main(args):
    clear_progress()

    with open(os.path.join("/data/input", args.input), "rt", encoding="utf-8") as fp:
        input_params = json.load(fp)

    try:
        save_progress({
            "status": "running",
            "stage": 0,
            "message": "Build and download pretrained model..."
        })
        mode = input_params["mode"]
        model_name = input_params["model_name"]
        model_path = os.path.join("/model", model_name, "model.pth")
        label_path = os.path.join("/model", model_name, "class_to_idx.json")
        data_path = os.path.join("/dataset", input_params["dataset"])
        resume_train = input_params.get("resume_train", False)
        batch_size = input_params.get("batch_size", 32)

        if mode == "train":
            max_epochs = input_params.get("epochs", 8)
            train_loader, test_loader, class_to_idx = load_data(data_path, batch_size)
            num_classes = len(class_to_idx.items())
            model = FaceNet(model_path, num_classes=num_classes, resume_train=resume_train)
            train_loss, train_acc, test_loss, test_acc = model.train(train_loader, test_loader, epochs=max_epochs)

            # save class idx info
            if label_path is not None:
                label_dir = os.path.dirname(label_path)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)
                with open(label_path, "wt", encoding="utf-8") as fp:
                    json.dump(class_to_idx, fp)

            save_result({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            }, args.output)
        elif mode == "unlearn":
            max_epochs = input_params.get("epochs", 2)
            print("TODO: unlearning")
    except KeyError as e:
        print("KeyError:", e)
        save_progress({
            "status": "done"
        })
        save_result({
            "mode": "train",
            "model_name": "my_model",
            "dataset": "dataset01"
        }, args.output)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
