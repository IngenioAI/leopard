import os
import json
import argparse
import warnings
import pprint

from target.utils import clear_progress, save_progress, save_result

from attack_runner import runner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


def attack(params):
    if "model_path" in params:
        args = argparse.Namespace(**params)
        pprint.pprint(args)
        return runner(args)
    else:
        return {
            "model_path": "/models/model00/best.h5",
            "train": False,
            "dp_on": False,
            "n_class": 10,
            "attack": "custom"
        }


def main(args):
    clear_progress()

    with open(os.path.join("/data/input", args.input), "rt", encoding="utf-8") as fp:
        input_params = json.load(fp)

    ret = attack(input_params)
    save_result(ret, args.output)
    save_progress({
        "status": "done"
    })


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
