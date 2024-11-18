import torch.nn as nn
import torch
from torch import optim
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import logging

from models.attack_models import MIAFC, SAMIA, SiameseNetwork2
from utils.visual_utils import plot_tsne

def get_attack_models(attack_type, n_classes):
    if attack_type == "nn":
        model = MIAFC(input_dim=n_classes, output_dim=2)
    elif attack_type == "samia":
        model = SAMIA(input_dim=n_classes*3, output_dim=2)
    elif attack_type == "nn_cls":
        model = MIAFC(input_dim=n_classes*2, output_dim=2)
    # elif attack_type == "pre":
    #     model = SiameseNetwork(input_dim=n_classes, output_dim=2)
    # elif attack_type == "new":
    #     model = SiameseNetwork2(input_dim=n_classes, output_dim=1)

    return model

# siamese network (ver: new)
class SiamAttacker2:
    def __init__(self, conf) -> None:
        self.conf = conf
        self.net = SiameseNetwork2(self.conf.n_classes)
        self.optim = optim.Adam(self.net.parameters(), weight_decay=self.conf.a_wd_, lr=self.conf.a_lr_)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optim,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
        self.criterion = nn.BCEWithLogitsLoss()

        self.save_log()

    def save_log(self):
        logging.basicConfig(filename='log.txt',
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger()

    def train(self, train_data, save_dir):
        losses = []
        self.net.to(self.conf.device)
        self.net.train()

        for epoch in range(self.conf.atk_epochs):
            loss = 0
            preds, labels, probs = np.array([]), np.array([])
            for vec1, vec2, label in train_data:
                vec1, vec2, label= vec1.to(self.conf.device), vec2.to(self.conf.device), label.to(self.conf.device)
                self.optim.zero_grad()
                output = self.net(vec1, vec2)
                loss_bce = self.criterion(output, label.float())

                loss += loss_bce.item()
                loss_bce.backward()
                self.optim.step()

                preds_batch = torch.round(torch.sigmoid(output.squeeze())).cpu().detach().numpy().astype(int)
                prob_batch = torch.sigmoid(output)
                prob_batch = prob_batch.cpu().detach().numpy()
                labels_batch = label.cpu().detach().numpy()

                preds = np.concatenate((preds, preds_batch), axis=-1)
                labels = np.concatenate((labels, labels_batch), axis=-1)
                probs = np.concatenate((probs, prob_batch), axis=-1)

            precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, preds, average='binary')
            acc  = metrics.accuracy_score(labels, preds)
            fpr, tpr, _ = metrics.roc_curve(labels, probs, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            print("\tEpoch:%5d, AvgLoss: %.6f, Acc: %.3f, "
                "Precision: %.3f, Recall: %.3f, F1: %.3f, AUC: %.3f" %
                (epoch, loss / len(train_data), acc, precision, recall, f1, auc))

            losses.append(loss/len(train_data))

            self.scheduler.step()


        plt.plot(range(self.conf.atk_epochs), losses)
        plt.savefig(f'{save_dir}/train_loss.png')
        plt.close()

        print(f'Saving model in {save_dir}/attack_model.pt')
        torch.save(self.net.state_dict(), f'{save_dir}/attack_model.pt')

    def test(self, test_data, save_dir, is_mem = None):
        acc, f1, auc, precision, recall, = 0, 0, 0, 0, 0
        preds, labels, probs = np.array([]), np.array([]), np.array([])

        self.net.to(self.conf.device)
        self.net.eval()
        with torch.no_grad():
            for vec1, vec2, label in test_data:
                vec1, vec2, label= vec1.to(self.conf.device), vec2.to(self.conf.device), label.to(self.conf.device)
                output = self.net(vec1, vec2)
                preds_batch = torch.round(torch.sigmoid(output)).cpu().detach().numpy()
                labels_batch = label.cpu().detach().numpy()

                prob_batch = torch.sigmoid(output)
                prob_batch = prob_batch.cpu().detach().numpy()

                probs = np.concatenate((probs, prob_batch), axis=-1)
                preds = np.concatenate((preds, preds_batch), axis=-1)
                labels = np.concatenate((labels, labels_batch), axis=-1)

        precision_batch, recall_batch, f1_batch, _ = metrics.precision_recall_fscore_support(labels, preds, average='binary')
        precision += precision_batch
        recall += recall_batch
        f1 += f1_batch

        acc += metrics.accuracy_score(labels, preds)
        fpr, tpr, _ = metrics.roc_curve(labels, probs, pos_label=1)
        auc += metrics.auc(fpr, tpr)

        print("Acc: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f, AUC: %.3f" %(acc, precision, recall, f1, auc))

        if is_mem is not None:
            if is_mem == True:
                fig_save_dir = f'{save_dir}/auc-real-test-mem.png'
            else:
                fig_save_dir = f'{save_dir}/auc-real-test-none-mem.png'
        else:
            fig_save_dir = f'{save_dir}/auc-test.png'

        plt.plot(fpr, tpr, label="Attack AUC = " + str(auc))
        plt.legend(loc=4)
        plt.savefig(fig_save_dir)
        plt.close()

    def load(self, model_path):
        self.net.load_state_dict(torch.load(model_path, map_location=f'cuda:{self.conf.device}'))

    def extract_forget_samples(self, test_data, is_mem=False):
        '''
            base assumption: attacker has non-member samples
            non-member -- non-member => 1
            non-member -- member ==> 0
        '''
        v1_idxs = []
        v2_idxs = []

        acc, f1, auc, precision, recall, = 0, 0, 0, 0, 0
        preds, labels, probs = np.array([]), np.array([]), np.array([])

        self.net.to(self.conf.device)
        self.net.eval()
        with torch.no_grad():
            for batch_idx, (v1, v2, label) in enumerate(test_data):
                v1, v1_idx = v1[:, :-1], v1[:, -1]
                v2, v2_idx = v2[:, :-1], v2[:, -1]

                v1, v2, label = v1.to(self.conf.device), v2.to(self.conf.device), label.to(self.conf.device)
                output = self.net(v1, v2)
                preds_batch = torch.round(torch.sigmoid(output)).cpu().detach().numpy()
                labels_batch = label.cpu().detach().numpy()
                prob_batch = torch.sigmoid(output)
                prob_batch = prob_batch.cpu().detach().numpy()

                correct_idx = np.where((preds_batch == 0) & (labels_batch == 0))[0]
                v1_idxs.extend(v1_idx[correct_idx].cpu().numpy())
                v2_idxs.extend(v2_idx[correct_idx].cpu().numpy())

                preds = np.concatenate((preds, preds_batch), axis=-1)
                labels = np.concatenate((labels, labels_batch), axis=-1)
                probs = np.concatenate((probs, prob_batch), axis=-1)

        precision_batch, recall_batch, f1_batch, _ = metrics.precision_recall_fscore_support(labels, preds, average='binary')
        precision += precision_batch
        recall += recall_batch
        f1 += f1_batch

        acc += metrics.accuracy_score(labels, preds)
        fpr, tpr, _ = metrics.roc_curve(labels, probs, pos_label=1)
        auc += metrics.auc(fpr, tpr)

        print("Acc: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f, AUC: %.3f" %(acc, precision, recall, f1, auc))

        class_ = np.unique(labels)
        asr= {}

        for c in class_:
            c_idx = np.where(labels == c)
            c_total = len(c_idx[0])
            c_corr = np.sum(preds[c_idx] == c)
            class_acc = c_corr / c_total if c_total > 0 else 0
            asr[c] = class_acc

        if is_mem:
            print("Class0 Acc: %.3f, Class1 Acc(ASR): %.3f" %(asr[0], asr[1]))
        else:
            print("Class0 Acc(ASR): %.3f, Class1 Acc: %.3f" %(asr[0], asr[1]))

        return v1_idxs, v2_idxs


class Attacker:
    def __init__(self, conf) -> None:
        self.conf = conf
        self.net = get_attack_models(conf.attack_type, conf.n_classes)
        self.optim = optim.Adam(self.net.parameters(), weight_decay=self.conf.a_wd_, lr=self.conf.a_lr_)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_data, save_dir):
        losses = []
        self.net.to(self.conf.device)
        self.net.train()
        for epoch in range(self.conf.atk_epochs):
            loss = 0
            preds, labels, probs = np.array([]), np.array([]), np.array([])
            # preds, labels = [], []
            # acc_batch_li = []
            for input, label in train_data:
                input, label = input.to(self.conf.device), label.to(self.conf.device)
                self.optim.zero_grad()
                output = self.net(input)
                loss = self.criterion(output, label)
                loss.backward()
                loss += loss.item()
                self.optim.step()

                _, preds_batch = output.max(1)
                preds_batch = preds_batch.cpu().detach().numpy()
                labels_batch = label.cpu().detach().numpy()
                loss = loss.cpu().detach().numpy()
                prob_batch = torch.softmax(output, dim=1)[:, 1]
                prob_batch = prob_batch.cpu().detach().numpy()

                probs = np.concatenate((probs, prob_batch), axis=-1)
                preds = np.concatenate((preds, preds_batch), axis=-1)
                labels = np.concatenate((labels, labels_batch), axis=-1)


            precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, preds, average='binary')
            acc  = metrics.accuracy_score(labels, preds)
            fpr, tpr, _ = metrics.roc_curve(labels, probs, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            print("\tEpoch:%5d, AvgLoss: %.6f, Acc: %.3f, "
                "Precision: %.3f, Recall: %.3f, F1: %.3f, AUC: %.3f" %
                (epoch, loss / len(train_data), acc, precision, recall, f1, auc))

            losses.append(loss/len(train_data))

        plt.plot(range(self.conf.atk_epochs), losses)
        plt.savefig(f'{save_dir}/train_loss.png')
        plt.close()

        print('Saving model in ', f"{save_dir}/attack_model.pt")
        torch.save(self.net.state_dict(), f"{save_dir}/attack_model.pt")

    def test(self, test_data, save_dir, is_mem = None, mu_test=False):
        acc, f1, auc, precision, recall, = 0, 0, 0, 0, 0
        preds, labels, probs = np.array([]), np.array([]), np.array([])
        logits = []

        self.net.to(self.conf.device)
        self.net.eval()
        with torch.no_grad():
            for batch in test_data:
                if len(batch) > 2:
                    input, label, idx = batch
                else:
                    input, label = batch

                input, label = input.to(self.conf.device), label.to(self.conf.device)
                output = self.net(input)

                _, preds_batch = output.max(1)
                preds_batch = preds_batch.cpu().detach().numpy()
                labels_batch = label.cpu().detach().numpy()
                prob_batch = torch.softmax(output, dim=1)[:, 1]
                prob_batch = prob_batch.cpu().detach().numpy()

                probs = np.concatenate((probs, prob_batch), axis=-1)
                preds = np.concatenate((preds, preds_batch), axis=-1)
                labels = np.concatenate((labels, labels_batch), axis=-1)
                logits.append(output.cpu().detach().numpy())

        #cifar100/googlenet -- samia-mu(ADV_IMP)-eval --> nan error
        if np.any(np.isnan(probs)):
            print("NaN found in probs, replacing NaN --> 0")
            probs = np.nan_to_num(probs, nan=0)

        logits = np.concatenate(logits, axis=0)
        # plot_tsne(logits, labels, save_dir)
        precision_batch, recall_batch, f1_batch, _ = metrics.precision_recall_fscore_support(labels, preds, average='binary')
        precision += precision_batch
        recall += recall_batch
        f1 += f1_batch

        acc += metrics.accuracy_score(labels, preds)
        fpr, tpr, _ = metrics.roc_curve(labels, probs, pos_label=1)
        auc += metrics.auc(fpr, tpr)

        print("Acc: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f, AUC: %.3f" %(acc, precision, recall, f1, auc))

        class_ = np.unique(labels)
        asr= {}

        for c in class_:
            c_idx = np.where(labels == c)
            c_total = len(c_idx[0])
            c_corr = np.sum(preds[c_idx] == c)
            class_acc = c_corr / c_total if c_total > 0 else 0
            asr[c] = class_acc

        if len(asr)==2:
            print("Class0 Acc: %.3f, Class1 Acc(ASR): %.3f" %(asr[0], asr[1]))
        else:
            print(f"prediction: {preds}, len:{np.count_nonzero(preds == 1)}")
            print(f"ground-truth: {labels}, len:{np.count_nonzero(labels == 1)}")
            print("Class1 Acc(ASR): %.3f" %(asr[1]))


        if mu_test:
            fig_save_dir = f'{save_dir}/auc-test-mu_ver.png'
        else:
            fig_save_dir = f'{save_dir}/auc-test.png'

        plt.plot(fpr, tpr, label="Attack AUC = " + str(auc))
        plt.legend(loc=4)
        plt.savefig(fig_save_dir)
        plt.close()

        return acc, auc

    def load(self, model_path):
        self.net.load_state_dict(torch.load(model_path, map_location=f'cuda:{self.conf.device}'))


    def extract_forget_samples(self, test_data):
        forget_samples_idxs = []
        acc, f1, auc, precision, recall, = 0, 0, 0, 0, 0
        preds, labels, probs = np.array([]), np.array([]), np.array([])
        logits = []

        self.net.to(self.conf.device)
        self.net.eval()
        with torch.no_grad():
            for batch_idx, (input, label, indices) in enumerate(test_data):
                input, label = input.to(self.conf.device), label.to(self.conf.device)
                output = self.net(input)

                _, preds_batch = output.max(1)
                preds_batch = preds_batch.cpu().detach().numpy()
                labels_batch = label.cpu().detach().numpy()

                correct_idx = np.where((preds_batch == 1) & (labels_batch == 1))[0]
                forget_samples_idxs.extend(indices[correct_idx].cpu().numpy())
                prob_batch = torch.softmax(output, dim=1)[:, 1]
                prob_batch = prob_batch.cpu().detach().numpy()

                probs = np.concatenate((probs, prob_batch), axis=-1)

                preds = np.concatenate((preds, preds_batch), axis=-1)
                labels = np.concatenate((labels, labels_batch), axis=-1)
                logits.append(output.cpu().detach().numpy())


        logits = np.concatenate(logits, axis=0)

        precision_batch, recall_batch, f1_batch, _ = metrics.precision_recall_fscore_support(labels, preds, average='binary')
        precision += precision_batch
        recall += recall_batch
        f1 += f1_batch

        acc += metrics.accuracy_score(labels, preds)
        fpr, tpr, _ = metrics.roc_curve(labels, probs, pos_label=1)
        auc += metrics.auc(fpr, tpr)

        print("Acc: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f, AUC: %.3f" %(acc, precision, recall, f1, auc))

        class_ = np.unique(labels)
        asr= {}

        for c in class_:
            c_idx = np.where(labels == c)
            c_total = len(c_idx[0])
            c_corr = np.sum(preds[c_idx] == c)
            class_acc = c_corr / c_total if c_total > 0 else 0
            asr[c] = class_acc

        print("Class0 Acc: %.3f, Class1 Acc(ASR): %.3f" %(asr[0], asr[1]))

        return forget_samples_idxs