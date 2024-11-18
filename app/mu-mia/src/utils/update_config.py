import argparse

import importlib
import os

def update_config_file(args):
    if args.n_classes == 10:
        if args.data_type == "kr_celeb":
            save_dir = f"kr_celeb_models"
        else:
            save_dir = f"cifar{args.n_classes}_models/{args.model_arch}_cifar{args.n_classes}"
    elif args.n_classes == 100:
        save_dir = f"cifar{args.n_classes}_models/{args.net}_cifar{args.n_classes}"

    os.makedirs(save_dir, exist_ok=True)

    with open('utils/config.py', 'r') as f:
        lines = f.readlines()

    with open(f"{save_dir}/config.py", 'w') as f:
        for line in lines:
            if line.startswith('device'):
                if args.device is not None:
                    line = f'device = {args.device}\n'
            elif line.startswith('model_arch'):
                if args.model_arch is not None:
                    line = f'model_arch = "{args.model_arch}"\n'
            elif line.startswith('net'):
                if args.net is not None:
                    line = f'net = "{args.net}"\n'
            elif line.startswith('n_classes'):
                line = f'n_classes = {args.n_classes}\n'
            elif line.startswith('data_type'):
                if args.data_type == "kr_celeb":
                    line = f'data_type = "{args.data_type}"\n'
                else:
                    line = f'data_type = f\'cifar{args.n_classes}\'\n'
            elif line.startswith('attack_type'):
                if args.attack_type is not None:
                    line = f'attack_type = "{args.attack_type}"\n'
            elif line.startswith('data_dir'):
                if args.n_classes is not None:
                    line = f'data_dir = f\'/miadata/datasets/cifar{{{args.n_classes}}}-data\'\n'
            elif line.startswith('save_dir'):
                line = f'save_dir = "/miadata/{save_dir}"\n'

            elif line.startswith('mode ='):
                line = f'mode = "{args.mode}"\n'

            f.write(line)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='GPU device to use (0 or 1)')
    parser.add_argument('--model_arch', type=str, default="", help='Model architecture for CIFAR10')
    parser.add_argument('--net', type=str, default="", help='Model architecture for CIFAR100')
    parser.add_argument('--n_classes', type=int, default=0, choices=[10, 100], help='Number of classes for dataset')
    parser.add_argument('--attack_type', type=str, default="", help='Attack type')
    parser.add_argument('--data_type', type=str, default="", help='data type') #kr_celeb
    parser.add_argument('--mode', type=str, default='', choices=['original', 'retrain', 'unlearn'])
    parser.add_argument("--forget_class_idx", type=int, default=9)

    args = parser.parse_args()
    print(f"Parsed arguments: {args}")

    return args


def set_config(args = None):
    args_v = vars(get_args())
    if args is not None:
        args_v.update(vars(args))
    args = argparse.Namespace(**args_v)
    update_config_file(args)
    conf = None
    if args.n_classes == 10:
        if args.data_type == "kr_celeb":
            conf = importlib.import_module("kr_celeb_models.config")
        else:
            if args.model_arch == "resnet18":
                conf = importlib.import_module("cifar10_models.resnet18_cifar10.config")
            elif args.model_arch == "densenet121":
                conf = importlib.import_module("cifar10_models.densenet121_cifar10.config")
            elif args.model_arch == "vgg11_bn":
                conf = importlib.import_module("cifar10_models.vgg11_bn_cifar10.config")
            elif args.model_arch == "mobilenet_v2":
                conf = importlib.import_module("cifar10_models.mobilenet_v2_cifar10.config")
            elif args.model_arch == "googlenet":
                conf = importlib.import_module("cifar10_models.googlenet_cifar10.config")
            elif args.model_arch == "inception_v3":
                conf = importlib.import_module("cifar10_models.inception_v3_cifar10.config")
    elif args.n_classes == 100:
        if args.net == "resnet18":
            conf = importlib.import_module("cifar100_models.resnet18_cifar100.config")
        elif args.net == "densenet121":
            conf = importlib.import_module("cifar100_models.densenet121_cifar100.config")
        elif args.net == "vgg11":
            conf = importlib.import_module("cifar100_models.vgg11_cifar100.config")
        elif args.net == "mobilenetv2":
            conf = importlib.import_module("cifar100_models.mobilenetv2_cifar100.config")
        elif args.net == "googlenet":
            conf = importlib.import_module("cifar100_models.googlenet_cifar100.config")
        elif args.net == "inceptionv3":
            conf = importlib.import_module("cifar100_models.inceptionv3_cifar100.config")

    return conf

