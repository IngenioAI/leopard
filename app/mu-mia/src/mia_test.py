import os
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from facenet_pytorch import InceptionResnetV1

from models.attacker import Attacker
from utils.update_config import set_config
from utils.dataset import get_dataset, get_probs, get_siam_attack_dataset

from unlearn_demo import get_dataloader, get_dataset_demo
from unlearn_sample_gen import forget_class_samples

def cw_forget_data_sampling(conf, target_train, target_test, unlearned_model, unlearned_model_path, forget_sample_list):
    '''
        if unlearned "target_train == member -> non-member"
                     "target_test == non-member -> non-member"
    '''
    mu_attack_testset = forget_class_samples(conf, target_train, target_test, is_both=False, cor_list=forget_sample_list)
    print( f"For class-wise unlearning, \n"
            f"Attack Test Size(for unlearned model): {len(mu_attack_testset)}\n"
            f"Number of samples correctly identified as members in the attack on the model before unlearning --- {len(mu_attack_testset)} samples\n"
            )

    attack_test_data, attack_test_labels = zip(*mu_attack_testset)
    attack_test_data = torch.stack([torch.tensor(data.clone().detach()) for data in attack_test_data])
    attack_test_labels = torch.tensor(attack_test_labels)

    attack_test_dataset = TensorDataset(
        attack_test_data, attack_test_labels, torch.tensor(forget_sample_list)
    )
    attack_test_loader = DataLoader(
        attack_test_dataset,
        batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True
    )

    _, mu_tr_acc, _, mu_tr_prob, mu_tr_sens, mu_tr_trgs, mu_tr_idx_li = get_probs(conf, unlearned_model, attack_test_loader, unlearned_model_path)
    print(mu_tr_trgs) #all_list == forget_class_id
    mu_tr_trgs = F.one_hot(mu_tr_trgs, num_classes=conf.n_classes).float()

    if conf.attack_type == "nn":
        attack_tt_data = mu_tr_prob
    elif conf.attack_type == "samia":
        attack_tt_data = torch.cat([mu_tr_prob, mu_tr_sens, mu_tr_trgs], dim=1) #[3000,10]
    elif conf.attack_type == "nn_cls":
        attack_tt_data = torch.cat([mu_tr_prob, mu_tr_trgs], dim=1)

    attack_tt_labels = torch.cat([torch.ones(mu_tr_prob.size(0))], dim=0).long()
    attack_tt_indices = torch.cat([torch.tensor(forget_sample_list)], dim=0)

    attack_test_dataset = TensorDataset(attack_tt_data, attack_tt_labels, attack_tt_indices)
    print(f"Forget Train Acc(target=={conf.forget_class_idx}): {mu_tr_acc}\n")

    print("Building Attack Dataset...")
    attack_test_loader = DataLoader(attack_test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)
    print(f"Attack Test Size: {len(attack_test_dataset)}")

    return attack_test_loader

def cw_utility_data_sampling(conf, target_train, target_test, unlearned_model, unlearned_model_path, cor_list):
    train_remain_forget_set, test_forget_set, train_retain_set, test_retain_set = forget_class_samples(conf, target_train, target_test, is_both=True, cor_list=cor_list)
    print( f"For class-wise unlearning, \n"
            f"Forget Samples Size (Non-member): {len(test_forget_set)}\n"
            f"Retain Samples Size (Member): {len(train_retain_set)}\n"
            f"Retain Samples Size (Non-member): {len(test_retain_set)}\n"
            )

    train_remain_forget_loader, test_forget_loader = get_dataloader(train_remain_forget_set, test_forget_set, conf)
    train_retain_set_loader, test_retain_set_loader = get_dataloader(train_retain_set, test_retain_set, conf)

    # 'tr' means target model(not shadow model)
    _, forget_tr_acc, _, forget_tr_prob, forget_tr_sens, forget_tr_trgs = get_probs(conf, unlearned_model, train_remain_forget_loader, unlearned_model_path)
    _, forget_tt_acc, _, forget_tt_prob, forget_tt_sens, forget_tt_trgs = get_probs(conf, unlearned_model, test_forget_loader, unlearned_model_path)

    _, retain_tr_acc, _, retain_tr_prob, retain_tr_sens, retain_tr_trgs = get_probs(conf, unlearned_model, train_retain_set_loader, unlearned_model_path)
    _, retain_tt_acc, _, retain_tt_prob, retain_tt_sens, retain_tt_trgs = get_probs(conf, unlearned_model, test_retain_set_loader, unlearned_model_path)

    forget_tr_probs = torch.cat([forget_tr_prob, forget_tt_prob], dim=0)
    forget_tr_sens = torch.cat([forget_tr_sens, forget_tt_sens], dim=0)
    forget_tr_trgs = torch.cat([forget_tr_trgs, forget_tt_trgs], dim=0)
    forget_tr_trgs = F.one_hot(forget_tr_trgs, num_classes=conf.n_classes).float() #[3000,10]

    retain_tr_probs = torch.cat([retain_tr_prob, retain_tt_prob], dim=0)
    retain_tr_sens = torch.cat([retain_tr_sens, retain_tt_sens], dim=0)
    retain_tr_trgs = torch.cat([retain_tr_trgs, retain_tt_trgs], dim=0)
    retain_tr_trgs = F.one_hot(retain_tr_trgs, num_classes=conf.n_classes).float() #[3000,10]


    if conf.attack_type == "nn":
        attack_tt_forget_data = forget_tr_probs
        attack_tt_retain_data = retain_tr_probs
    elif conf.attack_type == "samia":
        attack_tt_forget_data = torch.cat([forget_tr_probs, forget_tr_sens, forget_tr_trgs], dim=1) #[3000,10]
        attack_tt_retain_data = torch.cat([retain_tr_probs, retain_tr_sens, retain_tr_trgs], dim=1) #[3000,10]
    elif conf.attack_type == "nn_cls":
        attack_tt_forget_data = torch.cat([forget_tr_probs, forget_tr_trgs], dim=1)
        attack_tt_retain_data = torch.cat([retain_tr_probs, retain_tr_trgs], dim=1)
    elif conf.attack_type =="siamese":
        attack_tt_forget_data = get_siam_attack_dataset(forget_tr_prob, forget_tt_prob)
        attack_tt_retain_data = get_siam_attack_dataset(retain_tr_prob, retain_tt_prob)

    print(f"Loading Attack Models.. {conf.save_dir}")
    print(
            f"Forget Train Acc (Remaining member samples of forget samples): {forget_tr_acc}, \n"
            f"Forget Test Acc (Non-member samples of forget samples): {forget_tt_acc}, \n"
            f"Retain Train Acc (Non-member samples of retain samples): {retain_tr_acc}, \n"
            f"Retain Test Acc (Non-member samples of retain samples): {retain_tt_acc}, \n"
          )

    print("Building Attack Dataset...")
    if conf.attack_type == "pre" or conf.attack_type =="siamese":
        attack_test_forget_dataset =attack_tt_forget_data
        attack_test_retain_dataset =attack_tt_retain_data
    else:
        attack_tt_forget_labels = torch.cat([torch.ones(forget_tr_prob.size(0)), torch.zeros(forget_tt_prob.size(0))], dim=0).long()
        attack_test_forget_dataset = TensorDataset(attack_tt_forget_data, attack_tt_forget_labels)
        attack_tt_retain_labels = torch.cat([torch.ones(retain_tr_prob.size(0)), torch.zeros(retain_tt_prob.size(0))], dim=0).long()
        attack_test_retain_dataset = TensorDataset(attack_tt_retain_data, attack_tt_retain_labels)

    attack_test_forget_loader = DataLoader(attack_test_forget_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)
    attack_test_retain_loader = DataLoader(attack_test_retain_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)
    print(f"Attack Test Size (using forget samples): {len(attack_test_forget_dataset)}")
    print(f"Attack Test Size (using retain samples): {len(attack_test_retain_dataset)}")

    return attack_test_forget_loader, attack_test_retain_loader

#for class-wise unlearning
def cw_mu_test(conf):
    _, _, target_train, target_test = get_dataset_demo(conf)

    # in class-wise unlearning "forget_sample_list" means correct list
    forget_list_data_path = f"{conf.save_dir}/attack-{conf.attack_type}/unlearned_model_attack_test_samples.pkl"
    with open(forget_list_data_path, 'rb') as f:
        forget_sample_list= pickle.load(f)

    unlearned_model_path = f'{conf.save_dir}/target/VGGFace2_unlearn_model_forget{conf.forget_class_idx}.pth'
    trg_model = InceptionResnetV1(
                    classify=True,
                    pretrained='vggface2',
                    num_classes=conf.n_classes
                    )

    attack_mu_test_loader = cw_forget_data_sampling(conf, target_train, target_test, trg_model, unlearned_model_path, forget_sample_list)
    attack_test_forget_loader, attack_test_retain_loader = cw_utility_data_sampling(conf, target_train, target_test, trg_model, unlearned_model_path, forget_sample_list)

    result = {}
    if conf.attack_type == 'nn':
        print("-----------------------------------------------------------------")
        print("[NN attack testing]")
        attack_save_dir = f"{conf.save_dir}/attack-nn"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.load(f"{attack_save_dir}/attack_model.pt")
        print("Unlearning Evaluation(using forget dataset)")
        forget_acc, _ = attacker.test(attack_mu_test_loader, attack_save_dir, mu_test=True)
        print("Attack Utility Test(using retain dataset)")
        retain_acc, _ = attacker.test(attack_test_retain_loader, attack_save_dir, mu_test=True)
        print("Attack Utility Test(using remain forget dataset)")
        remain_acc, _ =attacker.test(attack_test_forget_loader, attack_save_dir, mu_test=True)
        print("-----------------------------------------------------------------")

    elif conf.attack_type == 'samia':
        print("-----------------------------------------------------------------")
        print("[Self-Attention attack (SAMIA) testing]")
        attack_save_dir = f"{conf.save_dir}/attack-samia"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.load(f"{attack_save_dir}/attack_model.pt")
        print("Unlearning Evaluation(using forget dataset)")
        forget_acc, _ = attacker.test(attack_mu_test_loader, attack_save_dir, mu_test=True)
        print("Attack Utility Test(using retain dataset)")
        retain_acc, _ = attacker.test(attack_test_retain_loader, attack_save_dir, mu_test=True)
        print("Attack Utility Test(using remain forget dataset)")
        remain_acc, _ = attacker.test(attack_test_forget_loader, attack_save_dir, mu_test=True)
        print("-----------------------------------------------------------------")

    elif conf.attack_type == 'nn_cls':
        print("-----------------------------------------------------------------")
        print("[Confidence-based Neural Network attack with ground-truth class (NNCls) attack testing]")
        attack_save_dir = f"{conf.save_dir}/attack-nn_cls"
        os.makedirs(attack_save_dir, exist_ok = True)
        attacker = Attacker(conf)
        attacker.load(f"{attack_save_dir}/attack_model.pt")
        print("Unlearning Evaluation(using forget dataset)")
        forget_acc, _ = attacker.test(attack_mu_test_loader, attack_save_dir, mu_test=True)
        print("Attack Utility Test(using retain dataset)")
        retain_acc, _ = attacker.test(attack_test_retain_loader, attack_save_dir, mu_test=True)
        print("Attack Utility Test(using remain forget dataset)")
        remain_acc, _ = attacker.test(attack_test_forget_loader, attack_save_dir, mu_test=True)
        print("-----------------------------------------------------------------")

    result["forget_acc"] = forget_acc
    result["retain_acc"] = retain_acc
    result["remain_acc"] = remain_acc
    return result

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    conf = set_config()
    print("Forget samples generation")
    cw_mu_test(conf)