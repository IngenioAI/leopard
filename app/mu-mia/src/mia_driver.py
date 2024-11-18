import os
import json
import argparse

from utils.update_config import set_config
from mia_test import cw_mu_test
from attack_test import attack_test
from unlearn_sample_gen import attack_test as generate_unlearn_sample
from attack_train import attack_train
from train import kr_celeb_model_train

def main(args):
    with open(f"/data/output/{args.output}", "wt", encoding="UTF-8") as fp:
        json.dump({}, fp, indent=4)

    with open(f"/data/input/{args.input}", "rt", encoding="utf-8") as fp:
        params = json.load(fp)
    print(params)

    op_mode = params["op_mode"]
    if op_mode == "train":
        params["mode"] = "original"
    else:
        params["mode"] = "unlearn"
    conf = set_config(argparse.Namespace(**params))
    if op_mode == "mia-unlearn":
        result = cw_mu_test(conf)
    elif op_mode == "unlearn-sample-generation":
        result = generate_unlearn_sample(conf)
    elif op_mode == "attack-test":
        result = attack_test(conf)
    elif op_mode == "attack-train":
        os.makedirs(conf.save_dir, exist_ok = True)
        result = attack_train(conf)
    elif op_mode in ["train", "unlearn"]:
        os.makedirs(conf.save_dir, exist_ok = True)
        result = kr_celeb_model_train(conf)
    else:
        print("Unknown op mode:", op_mode)

    with open(f"/data/output/{args.output}", "wt", encoding="UTF-8") as fp:
        if result is None:
            result = { "success": True, "message": "No result is good result"}
        json.dump(result, fp, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())