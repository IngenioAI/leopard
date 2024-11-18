
import torch
import torch.nn as nn
from pytorch_cifar100.utils import get_network, WarmUpLR
import torch.optim as optim
from pytorch_cifar10.module import CIFAR10Module


class TSTrainer:
    def __init__(self, conf):
        self.conf = conf

        if self.conf.data_type == 'cifar10':
            self.model = CIFAR10Module(self.conf.model_arch)
        elif self.conf.data_type == 'cifar100':
            self.model = get_network(net=self.conf.net)
        self.model.to(self.conf.device)

        self.criterion = nn.CrossEntropyLoss()
        
    # none, es, ls, cm
    def train(self, train_loader, test_loader, save_dir, is_target=False, log_pref=""):
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.conf.MILESTONES, gamma=0.2)

        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * self.conf.warm)
        early_params = 10
        ep_ch = 0
        best_epoch = 0
        best_acc = 0.0
        best_train_acc = 0.0
        best_model_weight = None
        for epoch in range(1, self.conf.max_epochs + 1):
            self.model.train()
            if epoch > self.conf.warm:
                    train_scheduler.step(epoch)
            correct = 0
            avg_loss = 0.0
            c = 0
            for batch in train_loader:
                
                images, labels = batch
                optimizer.zero_grad()
                labels, images = labels.to(self.conf.device), images.to(self.conf.device)
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum()
                avg_loss += loss.item()
                c += labels.size(0)

                if epoch <= self.conf.warm:  
                    warmup_scheduler.step()

            avg_loss = avg_loss / c
            train_acc = correct / c

            if log_pref:
                print("{}: Epoch: {:d}, Train Accuracy: {:.3f}, Average loss: {:.3f}".format(log_pref, epoch, train_acc, loss))

            acc = self.test(test_loader, epoch)
            if best_acc < acc:
                best_model_weight = self.model.state_dict()
                best_acc = acc
                best_train_acc = train_acc
                ep_ch = 0
                best_epoch = epoch
            else:
                ep_ch += 1
            
            if (ep_ch >= early_params) and (self.conf.attack_type=="es"):
                print(f"Early stopping at epoch {epoch}. Best accuracy: {best_acc:.3f}")
                break

        if best_model_weight is not None:
            if is_target:
                print(f"Best Epoch: {best_epoch}")
                torch.save(best_model_weight, f'{save_dir}/target_model.pt')
            else:
                print(f"Best Epoch: {best_epoch}")
                torch.save(best_model_weight, f'{save_dir}/shadow_model.pt')

        return best_train_acc, best_acc

    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        c = 0
        with torch.no_grad():
            for (images, labels) in test_loader:
                labels, images = labels.to(self.conf.device), images.to(self.conf.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum()
                c += labels.size(0)

        print('Evaluating Network.....')
        test_loss = test_loss / c
        avg_acc = correct / c
    
        print('Test set: Epoch: {}, Average loss: {:.4f}, Test Accuracy: {:.4f}'.format(
            epoch,
            test_loss,
            avg_acc,
        ))

        return correct/ len(test_loader.dataset)
