import warnings
import argparse
import pprint
import os

from attack_runner import runner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for attack configuration.')
    parser.add_argument('--model_path', default='target/weights/cifar10.h5',
                        help='Absolute path where pretrained model is saved')
    parser.add_argument('--train', default=False,
                        help='If no target model passed, train it with Cifar10 first')
    parser.add_argument('--dp_on', default=False,
                        help='Train with Differential Privacy (now only for Tensorflow models)')
    parser.add_argument('--n_class', default=10,
                        help='Number of classes of train data')
    parser.add_argument('--attack', default='custom',
                        help='Attack type: "custom" | "lira" | "population" | "reference" | "shadow" ')
    args = parser.parse_args()

    pprint.pprint(args)

    runner(args)
