import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch

from attacks.config import aconf, device
from _utils.data import TData


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


def load_torch_cifar(num_class=10):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    if num_class == 10:
        trainset = CIFAR10(root='./data', train=True,
                           download=True, transform=transform)

        testset = CIFAR10(root='./data', train=False,
                          download=True, transform=transform)
    else:
        trainset = CIFAR100(root='./data', train=True,
                            download=True, transform=transform)

        testset = CIFAR100(root='./data', train=False,
                           download=True, transform=transform)

    x, y = ds_to_numpy(trainset, testset)

    return TData(
        train_data=trainset.data,
        test_data=testset.data,
        train_labels=np.array(trainset.targets),
        test_labels=np.array(testset.targets),
        x_concat=x,
        y_concat=y
    )


def torch_predict(model, dataset):
    logits = []
    data_loader = DataLoader(dataset, batch_size=aconf['batch_size'])
    model.eval()

    for x in tqdm(data_loader):
        x = x.to(device)
        if x.shape[0] > 3:
            x = torch.transpose(x, 1, -1)

        pred = model(x.float())
        pred = pred.cpu().detach().numpy().copy()
        logits.append(pred)
    logits = np.concatenate(np.asarray(logits, dtype="object").copy(), axis=0)
    return logits


def torch_train(model, num_class, dataset=None, checkpoint_path=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=aconf['lr'])

    if dataset is None:
        dataset = load_torch_cifar(num_class)

    train_loader = DataLoader(group_data(
        dataset.train_data, dataset.train_labels), batch_size=aconf['batch_size'], shuffle=True)

    for epoch in range(aconf['epochs']):
        train_loss = 0
        train_acc = 0
        for x, y in tqdm(train_loader):
            x = torch.transpose(x, 1, -1).to(device)
            y = y.type(torch.LongTensor).to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            _, preds = torch.max(pred, 1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_acc += torch.sum(preds == y.data)

        print('epoch: {}, accuracy: {:.4f}, loss: {:.4f}'.format(epoch,
                                                                 train_acc /
                                                                 len(train_loader.dataset),
                                                                 train_loss / len(train_loader.dataset)))

    if checkpoint_path is not None:
        torch.save(model, checkpoint_path)
