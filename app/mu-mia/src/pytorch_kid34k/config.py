import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Membership inference Attacks on Network Pruning')
    parser.add_argument('--device', default=0, type=int, help="GPU id to use")
    parser.add_argument('--local_rank', default=0, type=int)
    # parser.add_argument('config_path', default=0, type=str, help="config file path")
    parser.add_argument('--dataset_name', default='cheapfake_ver2', type=str)
    # parser.add_argument('--train_data_json_path', default='/home/work/yujeong/pe2/target/cheapfake/data/splitted_ver2/trainval2.json', type=str)
    # parser.add_argument('--test_data_json_path', default='/home/work/yujeong/pe2/target/cheapfake/data/splitted_ver2/test2.json', type=str)

    # parser.add_argument('--s_train_data_json_path', default='/home/work/yujeong/pe2/mia_prune/reconstructed_img/genuine_both/cheapfake_ver2_AutoEncoder128/img_out/trainval2_Shadow.json', type=str)
    # parser.add_argument('--s_test_data_json_path', default='/home/work/yujeong/pe2/mia_prune/reconstructed_img/genuine_both/cheapfake_ver2_AutoEncoder128/img_out/test2_Shadow.json', type=str)
    # parser.add_argument('--v_train_data_json_path', default='/home/work/yujeong/pe2/mia_prune/reconstructed_img/genuine_both/cheapfake_ver2_AutoEncoder128/img_out/trainval2_Victim.json', type=str)
    # parser.add_argument('--v_test_data_json_path', default='/home/work/yujeong/pe2/mia_prune/reconstructed_img/genuine_both/cheapfake_ver2_AutoEncoder128/img_out/test2_Victim.json', type=str)

    #using_org_img
    parser.add_argument('--s_org_train_data_json_path', default='/home/yujeong/project/privacy/pe2/target/cheapfake/data/splitted_ver2/trainval2_Shadow.json', type=str)
    parser.add_argument('--s_org_test_data_json_path', default='/home/yujeong/project/privacy/pe2/target/cheapfake/data/splitted_ver2/test2_Shadow.json', type=str)
    parser.add_argument('--v_org_train_data_json_path', default='/home/yujeong/project/privacy/pe2/target/cheapfake/data/splitted_ver2/trainval2_Victim.json', type=str)
    parser.add_argument('--v_org_test_data_json_path', default='/home/yujeong/project/privacy/pe2/target/cheapfake/data/splitted_ver2/test2_Victim.json', type=str)

    parser.add_argument('--model_name_', default='resnet50', type=str)
    parser.add_argument('--num_cls', default=2, type=int)
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--image_size', default=28, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    # parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--early_stop', default=15, type=int)
    # parser.add_argument('--batch_size', default=128, type=int)
    

    parser.add_argument('--adaptive', action='store_true', help="use adaptive attack")
    parser.add_argument('--shadow_num', default=1, type=int)
    parser.add_argument('--defend', default='', type=str)
    parser.add_argument('--defend_arg', default=4, type=float)
    parser.add_argument('--attacks', default="samia", type=str)
    parser.add_argument('--original', action='store_true', help="original=true, then launch attack against original model")

    #### dynamic ####
    parser.add_argument("--use_DDP", action="store_true")
    parser.add_argument("--data_root_dir", "-drd", type=str, default="/home/data/T9_v6")  #### 이미지 폴더 선택, ex) /home/data/T9_v4
    parser.add_argument("--json_name", "-jn", type=str, default='')
    parser.add_argument("--augmentation_config", "-ac", type=int, default=20)  
    parser.add_argument("--augmentation_test_config", "-atc", type=int, default=5) 
    parser.add_argument("--img_H", type=int, default=512) 
    parser.add_argument("--img_W", type=int, default=800) 
    parser.add_argument("--use_cutmix", action="store_true")
    parser.add_argument("--use_cutout", action="store_true")
    parser.add_argument("--use_awp", action="store_true")
    parser.add_argument("--model_name", choices=["classifier", "classifier_patchCE", "classifier_auxAE"], default='classifier')
    parser.add_argument("--model_names", type=str, default='resnet50')   # resnet50, resnetv2_50x1_bitm_in21k
    parser.add_argument("--batch_size", "-bs", type=int, default=16)
    parser.add_argument("--accum_iter", "-ai", type=int, default=2)
    parser.add_argument("--use_3class", action="store_true")
    parser.add_argument("--except_class", default=None, help="real, screen, print에서 1개 빼서 비교.")
    parser.add_argument("--use_balsampler", action="store_true")    
    parser.add_argument("--save_name", type=str, default="best_reproduce0")

    #### dataset ####
    parser.add_argument("--n_fold", type=int, default=1)
    parser.add_argument("--test_data_name", default=["V_test_0", "V_test_1", "V_test_2", "V_test_3"])
    parser.add_argument("--aug_type", type=str, default="torchvision", choices=["torchvision", "albumentations"], help="heavy cpu usage with albumentations") 
    # parser.add_argument("--n_workers", type=int, default=3)
    parser.add_argument("--cutmix_prob", type=float, default=0.5)
    parser.add_argument("--cutmix_beta" ,type=float, default=1.0)
    parser.add_argument("--cutout_n_holes", type=int, default=1)
    parser.add_argument("--cutout_length", type=int, default=200)

    #### train & eval ####
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--criterion_name", type=str, default="ce", choices=["ce"])
    parser.add_argument("--optimizer_name", type=str, default="sgd", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.01, choices=[0.001, 0.01, 0.1, 0.3])
    parser.add_argument("--betas", default=(0.9,0.999))
    parser.add_argument("--scheduler_name", type=str, default="step", choices=["cosine", "step"])
    parser.add_argument("--steplr_step_size", type=int, default=15)
    parser.add_argument("--steplr_gamma", type=float, default=0.1)
    parser.add_argument("--cosinelr_T_0", type=int, default=10)
    parser.add_argument("--cosinelr_T_mult", type=int, default=1)
    parser.add_argument("--cosinelr_eta_min", type=float, default=1e-5)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--awp_warmup", type=int, default=2)
    parser.add_argument("--awp_lr", type=float, default=0.001)
    parser.add_argument("--awp_eps", type=float, default=0.001)
    parser.add_argument("--lambda_recon", type=float, default=10)

    #### model ####
    parser.add_argument("--no_pretrained", action="store_true")

    #### save ####
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_root_dir", type=str, default="outputs")

    #### config ####
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--attack_save_path", type=str)
    parser.add_argument("--org_sample_test", action="store_true")
    parser.add_argument("--t_sne", action="store_true")
    parser.add_argument("--d_print", action="store_true")
    parser.add_argument("--class_acc", action="store_true")
    parser.add_argument("--confusion_matrix", action="store_true")

    # 'print' label test
    parser.add_argument("--special_test_ver3", action="store_true") 

    parser.add_argument("--small_test", action="store_true")
    parser.add_argument("--print_test", action="store_true")
    parser.add_argument("--screen_test", action="store_true")
    parser.add_argument("--is_train_result", action="store_true")
    parser.add_argument("--ae_test", action="store_true")
    parser.add_argument("--use_org_img_for_am_test", action="store_true")
    parser.add_argument('--ae_model_name', default='cheapfake_ver2_AutoEncoder128', type=str)
    # parser.add_argument('--ae_model_dir_pth', default='pe2/mia_prune/reconstructed_img/genuine_both/cheapfake_ver2_AutoEncoder128', type=str)

    return parser.parse_args()