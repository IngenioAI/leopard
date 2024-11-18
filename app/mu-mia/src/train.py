import os
import torch.nn.functional as F
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_cifar10.module import CIFAR10Module
from utils.dataset import get_dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import shutil
import random
import torch.backends.cudnn as cudnn
from pytorch_kid34k.models.classifier import Classifier
import torch.nn as nn
import torch.optim as optim
from pytorch_cifar100.utils import get_network, WarmUpLR
from pytorch_kid34k.models.classifier import Classifier
from pytorch_kid34k.config import get_args
from utils.update_config import set_config
from unlearn_demo import main as kr_celeb_model_train
from models.ts_trainer import TSTrainer
import warnings

def target_shadow_dataset(conf):
    print(f"Loading Dataset... {conf.data_type}")
    train_dataset = get_dataset(conf, train=True)
    test_dataset = get_dataset(conf, train=False)

    total_dataset = ConcatDataset([train_dataset, test_dataset])
    total_size = len(total_dataset)

    target_list, shadow_list = train_test_split(list(range(total_size)), test_size=0.5, random_state=conf.seed)
    target_train_list, target_test_list = train_test_split(target_list, test_size=0.5, random_state=conf.seed)
    shadow_train_list, shadow_test_list = train_test_split(shadow_list, test_size=0.5, random_state=conf.seed)

    with open(f"{conf.save_dir}/data.pkl", 'wb') as f:
        pickle.dump([target_train_list, target_test_list, shadow_train_list, shadow_test_list], f)

    target_train_dataset = Subset(total_dataset, target_train_list)
    target_test_dataset = Subset(total_dataset, target_test_list)

    shadow_train_dataset = Subset(total_dataset, shadow_train_list)
    shadow_test_dataset = Subset(total_dataset, shadow_test_list)

    print(f"Total Data Size: {total_size}, "
          f"Target Train Size: {len(target_train_dataset)}, "
          f"Target Test Size: {len(target_test_dataset)}, "
          f"Shadow Train Size: {len(shadow_train_dataset)}, "
          f"Shadow Test Size: {len(shadow_test_dataset)}, ")

    return target_train_dataset, target_test_dataset, shadow_train_dataset, shadow_test_dataset

def get_save_dir(args, small_test=False):
    if small_test:
        if args.ae_test:
            if args.print_test:
                save_folder = f"kid34k_models_small/reconstructed_result/{args.ae_model_name}/results/genuine_print/{args.dataset_name}_{args.model_name_}"
            elif args.screen_test:
                save_folder = f"kid34k_models_small/reconstructed_result/{args.ae_model_name}/results/genuine_screen/{args.dataset_name}_{args.model_name_}"
            else:
                save_folder = f"kid34k_models_small/reconstructed_result/{args.ae_model_name}/results/genuine_both/{args.dataset_name}_{args.model_name_}"
        else:
            if args.print_test:
                save_folder = f"kid34k_models_small/genuine_print/{args.dataset_name}_{args.model_name_}"
            elif args.screen_test:
                save_folder = f"kid34k_models_small/genuine_screen/{args.dataset_name}_{args.model_name_}"
            else:
                save_folder = f"kid34k_models_small/genuine_both/{args.dataset_name}_{args.model_name_}"
    else:
        if args.ae_test:
            if args.print_test:
                save_folder = f"kid34k_models/reconstructed_result/{args.ae_model_name}/results/genuine_print/{args.dataset_name}_{args.model_name_}"
            elif args.screen_test:
                save_folder = f"kid34k_models/reconstructed_result/{args.ae_model_name}/results/genuine_screen/{args.dataset_name}_{args.model_name_}"
            else:
                save_folder = f"kid34k_models/reconstructed_result/{args.ae_model_name}/results/genuine_both/{args.dataset_name}_{args.model_name_}"
        else:
            if args.print_test:
                save_folder = f"kid34k_models/genuine_print/{args.dataset_name}_{args.model_name_}"
            elif args.screen_test:
                save_folder = f"kid34k_models/genuine_screen/{args.dataset_name}_{args.model_name_}"
            else:
                save_folder = f"kid34k_models/genuine_both/{args.dataset_name}_{args.model_name_}"
    return save_folder

def target_shadow_kid34k(small_test):
    args = get_args()
    args.local_rank = args.device
    args.n_gpus = 1

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cudnn.benchmark = True

    if small_test:
        if args.ae_test:
            if args.print_test:
                save_folder = f"kid34k_models_small/reconstructed_result/{args.ae_model_name}/results/genuine_print/{conf.data_type}_{conf.net}"
            elif args.screen_test:
                save_folder = f"kid34k_models_small/reconstructed_result/{args.ae_model_name}/results/genuine_screen/{conf.data_type}_{conf.net}"
            else:
                save_folder = f"kid34k_models_small/reconstructed_result/{args.ae_model_name}/results/genuine_both/{conf.data_type}_{conf.net}"
        else:
            if args.print_test:
                save_folder = f"kid34k_models_small/genuine_print/{conf.data_type}_{conf.net}"
            elif args.screen_test:
                save_folder = f"kid34k_models_small/genuine_screen/{conf.data_type}_{conf.net}"
            else:
                save_folder = f"kid34k_models_small/genuine_both/{conf.data_type}_{conf.net}"
    else:
        if args.ae_test:
            if args.print_test:
                save_folder = f"kid34k_models/reconstructed_result/{args.ae_model_name}/results/genuine_print/{conf.data_type}_{conf.net}"
            elif args.screen_test:
                save_folder = f"kid34k_models/reconstructed_result/{args.ae_model_name}/results/genuine_screen/{conf.data_type}_{conf.net}"
            else:
                save_folder = f"kid34k_models/reconstructed_result/{args.ae_model_name}/results/genuine_both/{conf.data_type}_{conf.net}"
        else:
            if args.print_test:
                save_folder = f"kid34k_models/genuine_print/{conf.data_type}_{conf.net}"
            elif args.screen_test:
                save_folder = f"kid34k_models/genuine_screen/{conf.data_type}_{conf.net}"
            else:
                save_folder = f"kid34k_models/genuine_both/{conf.data_type}_{conf.net}"
    

    print(f"Save Folder: {save_folder}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(f"{save_folder}/args.txt", "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    victim_train_dataset, victim_test_dataset, shadow_train_dataset, shadow_test_dataset = get_dataset(None, args, small_test=small_test)

    
    print(f"Total Data Size: {len(victim_train_dataset)+len(victim_test_dataset)+len(shadow_train_dataset)+len(shadow_test_dataset)}, \n"
          f"Victim Train Size: {len(victim_train_dataset)}, "
          f"Victim Test Size: {len(victim_test_dataset)}, \n"
          f"Shadow Train Size: {len(shadow_train_dataset)}, "
          f"Shadow Test Size: {len(shadow_test_dataset)}, ")

    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    
    print("Train Victim Model")
    victim_model_save_folder = save_folder + "/victim_model"
    os.makedirs(victim_model_save_folder, exist_ok = True)

    victim_model = Classifier(args)
    best_acc, best_train_acc, best_epoch = 0, 0, 0 

    for epoch in range(args.n_epochs):
        train_acc, train_loss = victim_model.train_one_epoch(victim_train_loader, f"Epoch {epoch} Train")
        test_acc, test_loss, _ = victim_model.test(victim_test_loader, f"Epoch {epoch} Test")
        if test_acc > best_acc:
            best_acc = test_acc
            best_train_acc = train_acc
            save_path = victim_model.save(victim_model_save_folder, test_acc, test_loss)
            best_path = save_path
            best_epoch = epoch

    shutil.copyfile(best_path, f"{victim_model_save_folder}/best.pth")
    print("Victim Model Eval")
    print(f"epoch: {best_epoch}, train accuracy: {best_train_acc}, test accuracy: {best_acc}")

    # Train shadow model
    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                        pin_memory=True)
    shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers,
                                    pin_memory=True)

    print(f"Train Shadow Model")
    shadow_model_save_folder = f"{save_folder}/shadow_model"
    os.makedirs(shadow_model_save_folder, exist_ok = True)
    
    shadow_model = Classifier(args)
    best_train_acc, best_acc, best_epoch = 0, 0, 0

    for epoch in range(args.n_epochs):
        train_acc, train_loss = shadow_model.train_one_epoch(shadow_train_loader, f"Epoch {epoch} Shadow Train")
        test_acc, test_loss, _ = shadow_model.test(shadow_test_loader, f"Epoch {epoch} Shadow Test")
        if test_acc > best_acc:
            best_acc = test_acc
            best_train_acc = train_acc
            save_path = shadow_model.save(shadow_model_save_folder, test_acc, test_loss)
            best_path = save_path
            best_epoch = epoch

    shutil.copyfile(best_path, f"{shadow_model_save_folder}/best.pth")
    print("Shadow Model Eval")
    print(f"epoch: {best_epoch}, train accuracy: {best_train_acc}, test accuracy: {best_acc}")


def target_shadow_cifar(conf):
    target_train_dataset, target_test_dataset, shadow_train_dataset, shadow_test_dataset = target_shadow_dataset(conf)

    target_train_loader = DataLoader(target_train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    target_save_dir = f"{conf.save_dir}/target"
    shadow_save_dir = f"{conf.save_dir}/shadow"

    os.makedirs(target_save_dir, exist_ok = True)
    os.makedirs(shadow_save_dir, exist_ok = True)

    trg_model = TSTrainer(conf)
    sh_model = TSTrainer(conf)

    if conf.n_classes == 10:
        m_name = conf.model_arch
    else:
        m_name = conf.net

    trg_train_acc, trg_test_acc = trg_model.train(target_train_loader, target_test_loader, target_save_dir, is_target=True, log_pref=f"[Traget Model Training(arch: {m_name}/cifar{conf.n_classes})]")
    sh_train_acc, sh_test_acc = sh_model.train(shadow_train_loader, shadow_test_loader, shadow_save_dir, is_target=False, log_pref=f"[Shadow Model Training(arch: {m_name}/cifar{conf.n_classes})]")

    print(f"Target Model Train Accuracy: {trg_train_acc}, Target Model Test Accuracy: {trg_test_acc}")
    print(f"Shadow Model Train Accuracy: {sh_train_acc}, Shadow Model Test Accuracy: {sh_test_acc}")
        
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore")
    conf = set_config()
    
    os.makedirs(conf.save_dir, exist_ok = True)
    if conf.data_type=='kr_celeb':
        kr_celeb_model_train(conf)
    elif conf.data_type=='kid34k':
        small_test=False #for debugging(not use)
        target_shadow_kid34k(small_test=small_test)
    else:
        target_shadow_cifar(conf)
