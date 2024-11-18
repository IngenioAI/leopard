import os, sys
from pytorch_cifar10.module import CIFAR10Module
from utils.dataset import get_dataset, get_probs, get_siam_attack_dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
import pickle
import torch.nn.functional as F
import torch
from facenet_pytorch import InceptionResnetV1
from models.attacker import SiamAttacker2, Attacker

from utils.update_config import set_config
from unlearn_demo import get_dataloader, get_dataset_demo
import warnings

def attack_test(conf):
    # args = get_args()
    # small_test = False
    # if conf.data_type == 'kid34k':
    #     kid34k_save_dir = get_save_dir(args, small_test=small_test)

    #     if small_test:
    #         target_train_dataset, target_test_dataset, shadow_train_dataset, shadow_test_dataset = get_dataset(train=True, args=args, save_folder=kid34k_save_dir, small_test=small_test)
    #     else:
    #         target_train_dataset, target_test_dataset, shadow_train_dataset, shadow_test_dataset = get_dataset(train=True, args=args, save_folder=kid34k_save_dir)

    #     trg_model_path = f'{kid34k_save_dir}/victim_model/best.pth'
    #     total_size = len(target_train_dataset) + len(target_test_dataset) + len(shadow_train_dataset) + len(shadow_test_dataset)
    # else:
    if conf.data_type =='kr_celeb':
        shadow_trainset, shadow_testset, target_trainset, target_testset = get_dataset_demo(conf)
        target_train_loader, target_test_loader = get_dataloader(target_trainset, target_testset, conf)

        trg_model_path = f'{conf.save_dir}/target/VGGFace2_original_target_model.pth'

        print(f"Total Data Size: {len(shadow_trainset) + len(shadow_testset) + len(target_trainset) + len(target_testset)}, "
            f"Target Train Size: {len(target_trainset)}, "
            f"Target Test Size: {len(target_testset)}, "
            f"Shadow Train Size: {len(shadow_trainset)}, "
            f"Shadow Test Size: {len(shadow_testset)}, ")

    else:
        train_dataset = get_dataset(conf, train=True)
        test_dataset = get_dataset(conf, train=False)

        total_dataset = ConcatDataset([train_dataset, test_dataset])
        total_size = len(total_dataset)

        data_path = f"{conf.save_dir}/data.pkl"
        with open(data_path, 'rb') as f:
            target_train_list, target_test_list, shadow_train_list, shadow_test_list = pickle.load(f)

        target_train_tensors = [total_dataset[i] for i in target_train_list]
        target_test_tensors = [total_dataset[i] for i in target_test_list]
        shadow_train_dataset = Subset(total_dataset, shadow_train_list)
        shadow_test_dataset = Subset(total_dataset, shadow_test_list)

        trg_model_path = f'{conf.save_dir}/target/target_model.pt'

        print(f"Total Data Size: {total_size}, "
            f"Target Train Size: {len(target_train_tensors)}, "
            f"Target Test Size: {len(target_test_tensors)}, "
            f"Shadow Train Size: {len(shadow_train_dataset)}, "
            f"Shadow Test Size: {len(shadow_test_dataset)}, ")

        target_train_data, target_train_labels = zip(*target_train_tensors)
        target_test_data, target_test_labels = zip(*target_test_tensors)

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

    # elif conf.data_type == 'kid34k':
    #     trg_model = Classifier(args)

    if conf.data_type == 'kr_celeb':
        _, trg_tr_acc, _, trg_tr_prob, trg_tr_sens, trg_tr_trgs = get_probs(conf, trg_model, target_train_loader, trg_model_path)
        _, trg_tt_acc, _, trg_tt_prob, trg_tt_sens, trg_tt_trgs = get_probs(conf, trg_model, target_test_loader, trg_model_path)
    else:
        _, trg_tr_acc, _, trg_tr_prob, trg_tr_sens, trg_tr_trgs, _ = get_probs(conf, trg_model, target_train_loader, trg_model_path)
        _, trg_tt_acc, _, trg_tt_prob, trg_tt_sens, trg_tt_trgs, _ = get_probs(conf, trg_model, target_test_loader, trg_model_path)

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
    elif conf.attack_type =="siamese":
        attack_tt_data = get_siam_attack_dataset(trg_tr_prob, trg_tt_prob)

    print(f"Loading Attack Models.. {conf.save_dir}")
    print(f"Target Model Train acc: {trg_tr_acc}, "
          f"Target Model Test acc: {trg_tt_acc}, ")

    print("Building Attack Dataset...")
    if conf.attack_type == "pre" or conf.attack_type =="siamese":
        attack_test_dataset =attack_tt_data
    else:
        attack_tt_labels = torch.cat([torch.ones(trg_tr_prob.size(0)), torch.zeros(trg_tt_prob.size(0))], dim=0).long()
        attack_test_dataset = TensorDataset(attack_tt_data, attack_tt_labels)

    attack_test_loader = DataLoader(attack_test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)
    print(f"Attack Test Size: {len(attack_test_dataset)}")

    # if conf.data_type == 'kid34k':
    #     attack_save_dir = kid34k_save_dir
    # else:
    #     attack_save_dir = save_dir

    if conf.attack_type == 'nn':
        print("-----------------------------------------------------------------")
        print("[NN attack testing]")
        attack_save_dir = f"{conf.save_dir}/attack-nn"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.load(f"{attack_save_dir}/attack_model.pt")
        acc, auc = attacker.test(attack_test_loader, attack_save_dir)
        print("-----------------------------------------------------------------")

    elif conf.attack_type == 'siamese':
        attack_save_dir = f"{conf.save_dir}/attack-siamese"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = SiamAttacker2(conf)
        attacker.load(f"{attack_save_dir}/attack_model.pt")

        print("-----------------------------------------------------------------")
        print("[Siamese attack testing]")
        attacker.test(attack_test_loader, attack_save_dir)
        print("[Siamese Realistic Testing - Attacker has a member sample]")
        attacker.test(attack_test_loader, attack_save_dir, is_mem=True)
        print("[Siamese Realistic Testing - Attacker has a non-member sample]")
        attacker.test(attack_test_loader, attack_save_dir, is_mem=False)
        print("-----------------------------------------------------------------")

    elif conf.attack_type == 'samia':
        print("-----------------------------------------------------------------")
        print("[Self-Attention attack (SAMIA) testing]")
        attack_save_dir = f"{conf.save_dir}/attack-samia"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.load(f"{attack_save_dir}/attack_model.pt")
        acc, auc = attacker.test(attack_test_loader, attack_save_dir)
        print("-----------------------------------------------------------------")

    elif conf.attack_type == 'nn_cls':
        print("-----------------------------------------------------------------")
        print("[Confidence-based Neural Network attack with ground-truth class (NNCls) attack testing]")
        attack_save_dir = f"{conf.save_dir}/attack-nn_cls"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.load(f"{attack_save_dir}/attack_model.pt")
        acc, auc = attacker.test(attack_test_loader, attack_save_dir)
        print("-----------------------------------------------------------------")

    return {
        "acc": acc,
        "auc": auc
    }

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    conf = set_config()
    attack_test(conf)
