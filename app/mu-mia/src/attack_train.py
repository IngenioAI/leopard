import os
import torch.nn.functional as F
import torch
from pytorch_cifar10.module import CIFAR10Module
from utils.dataset import get_dataset, get_probs, get_siam_attack_dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
import pickle
from models.attacker import SiamAttacker2, Attacker
from pytorch_cifar100.utils import get_network
# from pytorch_kid34k.config import get_args
from utils.update_config import set_config
from facenet_pytorch import InceptionResnetV1
from unlearn_demo import get_dataloader, get_dataset_demo
import warnings

def attack_train(conf):
    # if conf.data_type == 'kid34k':
    #     args = get_args()
    #     kid34k_save_dir = get_save_dir(args)
    #     target_train_dataset, target_test_dataset, shadow_train_dataset, shadow_test_dataset = get_dataset(train=True, args=args, save_folder=kid34k_save_dir)

    #     trg_model_path = f'{kid34k_save_dir}/victim_model/best.pth'
    #     sh_model_path = f'{kid34k_save_dir}/shadow_model/best.pth'
    #     total_size = len(target_train_dataset) + len(target_test_dataset) + len(shadow_train_dataset) + len(shadow_test_dataset)
    # else:

    if conf.data_type =='kr_celeb':
        shadow_trainset, shadow_testset, target_trainset, target_testset = get_dataset_demo(conf)
        shadow_train_loader, shadow_test_loader = get_dataloader(shadow_trainset, shadow_testset, conf)
        target_train_loader, target_test_loader = get_dataloader(target_trainset, target_testset, conf)
 
        trg_model_path = f'{conf.save_dir}/target/VGGFace2_original_target_model.pth'
        sh_model_path = f'{conf.save_dir}/shadow/VGGFace2_original_shadow_model.pth'

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

        target_train_dataset = Subset(total_dataset, target_train_list)
        target_test_dataset = Subset(total_dataset, target_test_list)
        shadow_train_dataset = Subset(total_dataset, shadow_train_list)
        shadow_test_dataset = Subset(total_dataset, shadow_test_list)

        trg_model_path = f'{conf.save_dir}/target/target_model.pt'
        sh_model_path = f'{conf.save_dir}/shadow/shadow_model.pt'


        print(f"Total Data Size: {total_size}, "
            f"Target Train Size: {len(target_train_dataset)}, "
            f"Target Test Size: {len(target_test_dataset)}, "
            f"Shadow Train Size: {len(shadow_train_dataset)}, "
            f"Shadow Test Size: {len(shadow_test_dataset)}, ")
        
        target_train_loader = DataLoader(target_train_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)
        target_test_loader = DataLoader(target_test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)
        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)


    if conf.data_type == 'cifar10':
        trg_model = CIFAR10Module(conf.model_arch)
        sh_model = CIFAR10Module(conf.model_arch)
    elif conf.data_type == 'cifar100':
        trg_model = get_network(conf.net)
        sh_model = get_network(conf.net)
    elif conf.data_type == 'kr_celeb':
        trg_model = InceptionResnetV1(
                    classify=True,
                    pretrained='vggface2',
                    num_classes=conf.n_classes
                    )
        sh_model = InceptionResnetV1(
                    classify=True,
                    pretrained='vggface2',
                    num_classes=conf.n_classes
                    )

    # elif conf.data_type == 'kid34k':
    #     trg_model = Classifier(args)
    #     sh_model = Classifier(args)

    _, trg_tr_acc, _, trg_tr_prob, trg_tr_sens, trg_tr_trgs = get_probs(conf, trg_model, target_train_loader, trg_model_path)
    _, trg_tt_acc, _, trg_tt_prob, trg_tt_sens, trg_tt_trgs = get_probs(conf, trg_model, target_test_loader, trg_model_path)

    _, sh_tr_acc, _, sh_tr_prob, sh_tr_sens, sh_tr_trgs = get_probs(conf, sh_model, shadow_train_loader, sh_model_path)
    _, sh_tt_acc, _, sh_tt_prob, sh_tt_sens, sh_tt_trgs = get_probs(conf, sh_model, shadow_test_loader, sh_model_path)

    sh_probs = torch.cat([sh_tr_prob, sh_tt_prob], dim=0)
    tr_probs = torch.cat([trg_tr_prob, trg_tt_prob], dim=0)
    sh_sens = torch.cat([sh_tr_sens, sh_tt_sens], dim=0)
    tr_sens = torch.cat([trg_tr_sens, trg_tt_sens], dim=0)
    sh_trgs = torch.cat([sh_tr_trgs, sh_tt_trgs], dim=0)
    tr_trgs = torch.cat([trg_tr_trgs, trg_tt_trgs], dim=0)

    sh_trgs = F.one_hot(sh_trgs, num_classes=conf.n_classes).float()
    tr_trgs = F.one_hot(tr_trgs, num_classes=conf.n_classes).float() #[3000,10]
    
    if conf.attack_type == "nn":
        attack_tr_data = sh_probs
        attack_tt_data = tr_probs
    elif conf.attack_type == "samia":
        attack_tr_data = torch.cat([sh_probs, sh_sens, sh_trgs], dim=1)
        attack_tt_data = torch.cat([tr_probs, tr_sens, tr_trgs], dim=1) #[3000,10]
    elif conf.attack_type == "nn_cls":
        attack_tr_data = torch.cat([sh_probs, sh_trgs], dim=1)
        attack_tt_data = torch.cat([tr_probs, tr_trgs], dim=1)
    elif conf.attack_type == "pre" or conf.attack_type =="siamese":
        attack_tr_data = get_siam_attack_dataset(sh_tr_prob, sh_tt_prob)
        attack_tt_data = get_siam_attack_dataset(trg_tr_prob, trg_tt_prob)
    
    print(f"Loading Target/Shadow Models.. {conf.save_dir}")
    print(f"Target Model Train acc: {trg_tr_acc}, "
          f"Target Model Test acc: {trg_tt_acc}, "
          f"Shadow Model Train acc: {sh_tr_acc}, "
          f"Shadow Model Test acc: {sh_tt_acc}, ")

    print("Building Attack Dataset...")

    if conf.attack_type == "pre" or conf.attack_type =="siamese":
        attack_train_dataset = attack_tr_data
        attack_test_dataset =attack_tt_data
    else:
        attack_tr_labels = torch.cat([torch.ones(sh_tr_prob.size(0)), torch.zeros(sh_tt_prob.size(0))], dim=0).long()
        attack_tt_labels = torch.cat([torch.ones(trg_tr_prob.size(0)), torch.zeros(trg_tt_prob.size(0))], dim=0).long()

        attack_train_dataset = TensorDataset(attack_tr_data, attack_tr_labels)
        attack_test_dataset = TensorDataset(attack_tt_data, attack_tt_labels)

    attack_train_loader = DataLoader(attack_train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, pin_memory=True)
    attack_test_loader = DataLoader(attack_test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)

    print(f"Attack Train Size: {len(attack_train_dataset)}, "
          f"Attack Test Size: {len(attack_test_dataset)}")
    
    # if conf.data_type == 'kid34k':
    #     attk_save_dir = kid34k_save_dir
    # else:
        # attk_save_dir = save_dir
        
    if conf.attack_type == 'nn':
        print("[NN attack testing]")
        attack_save_dir = f"{conf.save_dir}/attack-nn"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.train(attack_train_loader, attack_save_dir)
        attacker.test(attack_test_loader, attack_save_dir)

    elif conf.attack_type == 'samia':
        print("[Self-Attention attack (SAMIA) testing]")
        attack_save_dir = f"{conf.save_dir}/attack-samia"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.train(attack_train_loader, attack_save_dir)
        attacker.test(attack_test_loader, attack_save_dir)

    elif conf.attack_type == 'nn_cls':
        print("[Confidence-based Neural Network attack with ground-truth class (NNCls) attack testing]")
        attack_save_dir = f"{conf.save_dir}/attack-nn_cls"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.train(attack_train_loader, attack_save_dir)
        attacker.test(attack_test_loader, attack_save_dir)

    elif conf.attack_type == 'siamese':
        print("[Siamese attack testing]")
        attack_save_dir = f"{conf.save_dir}/attack-siamese"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = SiamAttacker2(conf)
        attacker.train(attack_train_loader, attack_save_dir)
        attacker.test(attack_test_loader, attack_save_dir)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore")
    conf = set_config()
    os.makedirs(conf.save_dir, exist_ok = True)

    attack_train(conf)


