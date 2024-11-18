import os, sys
from pytorch_cifar10.module import CIFAR10Module
from utils.dataset import get_dataset, get_probs
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
import pickle
import torch.nn.functional as F
import torch

from facenet_pytorch import InceptionResnetV1

from models.attacker import Attacker

from pytorch_cifar100.utils import get_network
from torchvision.transforms import ToPILImage

# from pytorch_kid34k.models.classifier import Classifier
# from pytorch_kid34k.data_utils.dataset import get_dataset as get_real_dataset
# from pytorch_kid34k.config import get_args

# from train import get_save_dir
# from models.attacker import MiaAttack
from utils.update_config import set_config
from unlearn_demo import get_dataset_demo, split_class_data
import warnings
# for kr_celeb test
# def save_sample_idx(conf, trainset, testset):
#     sample_idx = {
#         'train_idx': list(range(len(trainset))),
#         'test_idx': list(range(len(trainset), len(trainset) + len(testset)))
#     }

#     with open(f"{conf.save_dir}/{conf.attack_type}/target_forget_dataset.pkl", 'wb') as f:
#         pickle.dump(sample_idx, f)
#     return sample_idx['train_idx'], sample_idx['test_idx']



def forget_class_samples(conf, train_set, test_set, is_both=False, cor_list=None):
    '''
        1)  To get the index numbers of 'forget_samples' and 'retain_samples'.
            Create a list('cor_list') of indices for samples that were correctly predicted as members (on original model).
        2)  If 'cor_list' is not None, create a test set for the unlearned model attack by comparing 'cor_list' with forget_list.
    '''
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
    test_forget_set = Subset(test_set, test_forget_indices)

    if is_both: # Attack on the unlearned model (for utility)
        cor_set = set(cor_list)
        train_forget_remain_indices = [idx for idx in train_remain_indices if idx not in cor_set] # Forget sample indices not in cor_list
        train_forget_remain_set = Subset(train_set, train_forget_remain_indices)
        train_retain_set = Subset(train_set, train_remain_indices)
        test_retain_set = Subset(test_set, test_remain_indices)
        return train_forget_remain_set, test_forget_set, train_retain_set, test_retain_set
    else:
        if cor_list is not None:
            '''
                Attack on the unlearned model (for unleanierng eval)
                Samples correctly identified as members in the attack on the original model.
            '''
            mu_attack_test_set = Subset(train_set, cor_list)
            return mu_attack_test_set
        else:
            '''
                Attack on the original model (for unleanierng eval)
                To make cor_list.
            '''
            return train_forget_set, test_forget_set, train_forget_indices, test_forget_indices


def attack_test(conf):
    if conf.data_type =='kr_celeb':
        _, _, target_train_all, target_test_all = get_dataset_demo(conf)
        target_train, target_test, target_train_list, target_test_list = forget_class_samples(conf, target_train_all, target_test_all) # forget samples: member, non-member (target==forget_class_idx)
        # target_train_list, target_test_list = save_sample_idx(conf, target_train, target_test)
        trg_model_path = f"{conf.save_dir}/target/{conf.dataset}_original_target_model.pth"

        print(
            f"For class-wise unlearning, "
            f"Target Train Size(=forget samples(member)): {len(target_train)}, "
            f"Target Test Size(=forget samples(non-member)): {len(target_test)}, "
            )
    else:
        train_dataset = get_dataset(conf, train=True)
        test_dataset = get_dataset(conf, train=False)

        total_dataset = ConcatDataset([train_dataset, test_dataset])
        total_size = len(total_dataset)

        data_path = f"{conf.save_dir}/data.pkl"
        with open(data_path, 'rb') as f:
            target_train_list, target_test_list, _, _ = pickle.load(f)

        target_train = [total_dataset[i] for i in target_train_list]
        target_test = [total_dataset[i] for i in target_test_list]

        trg_model_path = f'{conf.save_dir}/target/target_model.pt'

        print(f"For instacne-wise unlearning, "
            f"Target Train Size: {len(target_train)}, "
            f"Target Test Size: {len(target_test)}, ")

    target_train_data, target_train_labels = zip(*target_train)
    target_test_data, target_test_labels = zip(*target_test)

    target_train_data = torch.stack([torch.tensor(data.clone().detach()) for data in target_train_data])
    target_train_labels = torch.tensor(target_train_labels)
    target_test_data = torch.stack([torch.tensor(data.clone().detach()) for data in target_test_data])
    target_test_labels = torch.tensor(target_test_labels)

    target_train_dataset = TensorDataset(
        target_train_data, target_train_labels, torch.tensor(target_train_list)
    )
    target_test_dataset = TensorDataset(
        target_test_data, target_test_labels, torch.tensor(target_test_list)
    )

    target_train_loader = DataLoader(
        target_train_dataset,
        batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True
    )

    target_test_loader = DataLoader(
        target_test_dataset,
        batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True
    )

    if conf.data_type == 'cifar10':
        trg_model = CIFAR10Module(conf.model_arch)
    elif conf.data_type == 'cifar100':
        trg_model = get_network(conf.net)
    elif conf.data_type == 'kr_celeb':
        trg_model = InceptionResnetV1(
                    classify=True,
                    pretrained='vggface2',
                    num_classes=conf.n_classes
                    )

    _, trg_tr_acc, _, trg_tr_prob, trg_tr_sens, trg_tr_trgs, trg_tr_idx_li = get_probs(conf, trg_model, target_train_loader, trg_model_path)
    _, trg_tt_acc, _, trg_tt_prob, trg_tt_sens, trg_tt_trgs, trg_tt_idx_li = get_probs(conf, trg_model, target_test_loader, trg_model_path)

    tr_probs = torch.cat([trg_tr_prob, trg_tt_prob], dim=0)
    tr_sens = torch.cat([trg_tr_sens, trg_tt_sens], dim=0)
    tr_trgs = torch.cat([trg_tr_trgs, trg_tt_trgs], dim=0)
    tr_trgs = F.one_hot(tr_trgs, num_classes=conf.n_classes).float() #[3000,10]

    if conf.attack_type == "nn":
        attack_tt_data = tr_probs
    elif conf.attack_type == "samia":
        attack_tt_data = torch.cat([tr_probs, tr_sens, tr_trgs], dim=1) #[3000,10]
    elif conf.attack_type == "nn_cls":
        attack_tt_data = torch.cat([tr_probs, tr_trgs], dim=1)

    attack_tt_labels = torch.cat([torch.ones(trg_tr_prob.size(0)), torch.zeros(trg_tt_prob.size(0))], dim=0).long()
    attack_tt_indices = torch.cat([trg_tr_idx_li, trg_tt_idx_li], dim=0)

    ## for instance-wise unlearning debugging
    # has_duplicates = len(attack_tt_indices) != len(torch.unique(attack_tt_indices))
    # if has_duplicates:
    #     print("has")
    # else:
    #     print("none")

    attack_test_dataset = TensorDataset(attack_tt_data, attack_tt_labels, attack_tt_indices)
    print(f"Target Model Train acc: {trg_tr_acc}, \n"
          f"Target Model Test acc: {trg_tt_acc}, \n")

    print("Building Attack Dataset...")
    attack_test_loader = DataLoader(attack_test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)
    print(f"Attack Test Size: {len(attack_test_dataset)}")

    if conf.attack_type == 'nn':
        print("-----------------------------------------------------------------")
        print("[NN attack testing]")
        if conf.data_type == "kr_celeb":
            print(f"Attack Test(target class = {conf.forget_class_idx})")
        attack_save_dir = f"{conf.save_dir}/attack-nn"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.load(f"{attack_save_dir}/attack_model.pt")
        forget_samples = attacker.extract_forget_samples(attack_test_loader)
        print("-----------------------------------------------------------------")

    elif conf.attack_type == 'samia':
        print("-----------------------------------------------------------------")
        print("[Self-Attention attack (SAMIA) testing]")
        if conf.data_type == "kr_celeb":
            print(f"Attack Test(target class = {conf.forget_class_idx})")
        attack_save_dir = f"{conf.save_dir}/attack-samia"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.load(f"{attack_save_dir}/attack_model.pt")
        forget_samples = attacker.extract_forget_samples(attack_test_loader)
        print("-----------------------------------------------------------------")

    elif conf.attack_type == 'nn_cls':
        print("-----------------------------------------------------------------")
        print("[Confidence-based Neural Network attack with ground-truth class (NNCls) attack testing]")
        if conf.data_type == "kr_celeb":
            print(f"Attack Test(target class = {conf.forget_class_idx})")
        attack_save_dir = f"{conf.save_dir}/attack-nn_cls"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.load(f"{attack_save_dir}/attack_model.pt")
        forget_samples = attacker.extract_forget_samples(attack_test_loader)
        print("-----------------------------------------------------------------")

    print("forget sampels: ", forget_samples)

    #for class-wise unlearning
    if conf.data_type =="kr_celeb":
        print(f"Saving unlearning evaluation sample list... {attack_save_dir}/unlearned_model_attack_test_samples.pkl\n")
        print(f"Evaluation sampels: {len(forget_samples)}")
        with open(f"{attack_save_dir}/unlearned_model_attack_test_samples.pkl", 'wb') as f:
            pickle.dump(forget_samples, f)

    #for instance-wise unlearning
    else:
        print(f"Saving forget sample list... {attack_save_dir}/forget_samples.pkl")
        with open(f"{attack_save_dir}/forget_samples.pkl", 'wb') as f:
            pickle.dump(forget_samples, f)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    conf = set_config()
    print("Forget samples generation")
    attack_test(conf)
