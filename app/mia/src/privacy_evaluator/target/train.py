import os
import json
from torch_target_defense import run
import argparse

from utils import clear_progress, save_result

def train(params):
    if "model_name" in params:
        args = argparse.Namespace(**params)
        return run(args)
    else:
        return {
            "model_name": "resnet20",
            "datasets": "cifar100",
            "defense": "none",
            "batch_size": 128,
            "num_workers": 5,
            "epochs": 300,
            "shadow_num": 1,
            "early_stop": 5,
            "weight_decay": 1e-4,
            "lr": 0.1,
            "momentum": 0.9,
            "seed": 1000
        }


def main(args):
    clear_progress()

    with open(os.path.join("/data/input", args.input), "rt", encoding="utf-8") as fp:
        input_params = json.load(fp)

    ret = train(input_params)
    save_result(ret, args.output)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())