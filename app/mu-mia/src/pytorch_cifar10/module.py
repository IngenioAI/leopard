import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy
import torch.nn.functional as F

from pytorch_cifar10.cifar10_models.densenet import densenet121, densenet161, densenet169
from pytorch_cifar10.cifar10_models.googlenet import googlenet
from pytorch_cifar10.cifar10_models.inception import inception_v3
from pytorch_cifar10.cifar10_models.mobilenetv2 import mobilenet_v2
from pytorch_cifar10.cifar10_models.resnet import resnet18, resnet34, resnet50
from pytorch_cifar10.cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from pytorch_cifar10.schduler import WarmupCosineLR
from mu.L2UL.utils import NormalizeLayer

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
}

class CIFAR10Module(pl.LightningModule):
    def __init__(self, model_arch):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.model = all_classifiers[model_arch]

    def forward(self, batch):
        if len(batch)>=2 and len(batch) < 5: #need fix(must batch size > 5)
            images, labels = batch
            predictions = self.model(images)
            loss = self.criterion(predictions, labels)
            accuracy = self.accuracy(predictions, labels)
            return loss, accuracy * 100, predictions
        else:
            predictions = self.model(batch)
            return predictions
     

    def training_step(self, batch, batch_nb):
        loss, accuracy, _ = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy, _ = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy, pred = self.forward(batch)
        self.log("acc/test", accuracy)
        return loss, accuracy, pred
    