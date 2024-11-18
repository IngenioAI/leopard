# import argparse
# import numpy as np
# import os
# import pickle
# import shutil
# import random
# import torch
# import torch.backends.cudnn as cudnn
# from sklearn.model_selection import train_test_split
# from torch.utils.data import ConcatDataset, DataLoader, Subset

# from pytorch_kid34k.config import get_args
# from pytorch_kid34k.data_utils.dataset import get_dataset
# from pytorch_kid34k.models.classifier import Classifier

# import utils.config as conf

# def main(small_test=False):
#     args = get_args()
#     args.local_rank = args.device
#     args.n_gpus = 1

#     torch.manual_seed(args.seed)
#     random.seed(args.seed)
#     np.random.seed(args.seed)

#     cudnn.benchmark = True

#     if small_test:
#         if args.ae_test:
#             if args.print_test:
#                 save_folder = f"kid34k_models_small/reconstructed_result/{args.ae_model_name}/results/genuine_print/{conf.data_type}_{conf.net}"
#             elif args.screen_test:
#                 save_folder = f"kid34k_models_small/reconstructed_result/{args.ae_model_name}/results/genuine_screen/{conf.data_type}_{conf.net}"
#             else:
#                 save_folder = f"kid34k_models_small/reconstructed_result/{args.ae_model_name}/results/genuine_both/{conf.data_type}_{conf.net}"
#         else:
#             if args.print_test:
#                 save_folder = f"kid34k_models_small/genuine_print/{conf.data_type}_{conf.net}"
#             elif args.screen_test:
#                 save_folder = f"kid34k_models_small/genuine_screen/{conf.data_type}_{conf.net}"
#             else:
#                 save_folder = f"kid34k_models_small/genuine_both/{conf.data_type}_{conf.net}"
#     else:
#         if args.ae_test:
#             if args.print_test:
#                 save_folder = f"kid34k_models/reconstructed_result/{args.ae_model_name}/results/genuine_print/{conf.data_type}_{conf.net}"
#             elif args.screen_test:
#                 save_folder = f"kid34k_models/reconstructed_result/{args.ae_model_name}/results/genuine_screen/{conf.data_type}_{conf.net}"
#             else:
#                 save_folder = f"kid34k_models/reconstructed_result/{args.ae_model_name}/results/genuine_both/{conf.data_type}_{conf.net}"
#         else:
#             if args.print_test:
#                 save_folder = f"kid34k_models/genuine_print/{conf.data_type}_{conf.net}"
#             elif args.screen_test:
#                 save_folder = f"kid34k_models/genuine_screen/{conf.data_type}_{conf.net}"
#             else:
#                 save_folder = f"kid34k_models/genuine_both/{conf.data_type}_{conf.net}"
    

#     print(f"Save Folder: {save_folder}")
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)

#     with open(f"{save_folder}/args.txt", "w") as f:
#         for arg in vars(args):
#             f.write(f"{arg}: {getattr(args, arg)}\n")

#     if args.ae_test:
#         S_trainset, S_validset, S_testset = get_dataset(args.s_train_data_json_path, args.s_test_data_json_path, args.print_test, args.screen_test, small_test=small_test)
#         V_trainset, V_validset, V_testset = get_dataset(args.v_org_train_data_json_path, args.v_org_test_data_json_path, args.print_test, args.screen_test, small_test=small_test)
#         print(f"Shadow train size: {len(S_trainset)}, Shadow valid size: {len(S_validset)}, Shadow test size: {len(S_testset)}")
#         print(f"Victim train size: {len(V_trainset)}, Victim valid size: {len(V_validset)}, Victim test size: {len(V_testset)}")
#     else:
#         S_trainset, S_validset, S_testset = get_dataset(args.s_org_train_data_json_path, args.s_org_test_data_json_path, args.print_test, args.screen_test, small_test=small_test)
#         V_trainset, V_validset, V_testset = get_dataset(args.v_org_train_data_json_path, args.v_org_test_data_json_path, args.print_test, args.screen_test, small_test=small_test)
#         print(f"Shadow train size: {len(S_trainset)}, Shadow valid size: {len(S_validset)}, Shadow test size: {len(S_testset)}")
#         print(f"Victim train size: {len(V_trainset)}, Victim valid size: {len(V_validset)}, Victim test size: {len(V_testset)}")
    
    
#     data_index_path = f"{save_folder}/data_index.pkl"
            
#     if args.dataset_name == 'cheapfake_ver1':
#         V_total_dataset = ConcatDataset([V_trainset, V_validset, V_testset])
#         S_total_dataset = ConcatDataset([S_trainset, S_validset, S_testset])

#         total_size = len(V_total_dataset) + len(S_total_dataset)

#         victim_train_list, victim_test_list = train_test_split(list(range(len(V_total_dataset))), test_size=0.5, random_state=args.seed)
#         shadow_train_list, shadow_test_list = train_test_split(list(range(len(S_total_dataset))), test_size=0.5, random_state=args.seed)

#         victim_train_dataset = Subset(V_total_dataset, victim_train_list)
#         victim_test_dataset = Subset(V_total_dataset, victim_test_list)
#         shadow_train_dataset = Subset(S_total_dataset, shadow_train_list)
#         shadow_test_dataset = Subset(S_total_dataset, shadow_test_list)

#         with open(data_index_path, 'wb') as f:
#             pickle.dump([victim_train_list, victim_test_list, shadow_train_list, shadow_test_list], f)

#     else:
#         V_total_dataset =  ConcatDataset([V_trainset, V_validset])
#         S_total_dataset =  ConcatDataset([S_trainset, S_validset])

#         total_size = len(V_total_dataset) + len(S_total_dataset) + len(V_testset) + len(S_testset)

#         victim_train_dataset = V_total_dataset
#         victim_test_dataset = V_testset
#         shadow_train_dataset = S_total_dataset
#         shadow_test_dataset = S_testset

    
#     print(f"Total Data Size: {total_size}, \n"
#           f"Victim Train Size: {len(victim_train_dataset)}, "
#           f"Victim Test Size: {len(victim_test_dataset)}, \n"
#           f"Shadow Train Size: {len(shadow_train_dataset)}, "
#           f"Shadow Test Size: {len(shadow_test_dataset)}, ")

#     victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
#     victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    
#     print("Train Victim Model")
#     victim_model_save_folder = save_folder + "/victim_model"
#     os.makedirs(victim_model_save_folder, exist_ok = True)

#     victim_model = Classifier(args)
#     best_acc, best_train_acc, best_epoch = 0, 0, 0 

#     for epoch in range(args.n_epochs):
#         train_acc, train_loss = victim_model.train_one_epoch(victim_train_loader, f"Epoch {epoch} Train")
#         test_acc, test_loss, _ = victim_model.test(victim_test_loader, f"Epoch {epoch} Test")
#         if test_acc > best_acc:
#             best_acc = test_acc
#             best_train_acc = train_acc
#             save_path = victim_model.save(victim_model_save_folder, test_acc, test_loss)
#             best_path = save_path
#             best_epoch = epoch

#     shutil.copyfile(best_path, f"{victim_model_save_folder}/best.pth")
#     print("Victim Model Eval")
#     print(f"epoch: {best_epoch}, train accuracy: {best_train_acc}, test accuracy: {best_acc}")

#     # Train shadow model
#     shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
#                                         pin_memory=True)
#     shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers,
#                                     pin_memory=True)

#     print(f"Train Shadow Model")
#     shadow_model_save_folder = f"{save_folder}/shadow_model"
#     os.makedirs(shadow_model_save_folder, exist_ok = True)
    
#     shadow_model = Classifier(args)
#     best_train_acc, best_acc, best_epoch = 0, 0, 0

#     for epoch in range(args.n_epochs):
#         train_acc, train_loss = shadow_model.train_one_epoch(shadow_train_loader, f"Epoch {epoch} Shadow Train")
#         test_acc, test_loss, _ = shadow_model.test(shadow_test_loader, f"Epoch {epoch} Shadow Test")
#         if test_acc > best_acc:
#             best_acc = test_acc
#             best_train_acc = train_acc
#             save_path = shadow_model.save(shadow_model_save_folder, test_acc, test_loss)
#             best_path = save_path
#             best_epoch = epoch

#     shutil.copyfile(best_path, f"{shadow_model_save_folder}/best.pth")
#     print("Shadow Model Eval")
#     print(f"epoch: {best_epoch}, train accuracy: {best_train_acc}, test accuracy: {best_acc}")

