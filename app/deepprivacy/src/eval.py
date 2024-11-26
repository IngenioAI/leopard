import os
import json
import argparse

def main(args):
    with open(f"/data/input/{args.input}", "rt", encoding="utf-8") as fp:
        params = json.load(fp)

    # load pre-evaluated result
    with open("metrics/result.json", "rt", encoding="utf-8") as fp:
        result = json.load(fp)

    with open(f"/data/output/{args.output}", "w") as f:
        json.dump(result, f)

    print(result)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
