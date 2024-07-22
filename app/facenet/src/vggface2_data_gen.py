import os
import argparse
import json
import csv
import random
import shutil

import torch, PIL
from facenet_pytorch import MTCNN

from utils import save_result, clear_progress, save_progress


def detect_face(mtcnn, source_path, target_train_path, target_test_path, test_split=0.0):
    for req_path in [target_train_path, target_test_path]:
        if not os.path.exists(req_path):
            os.makedirs(req_path)
    for name in os.listdir(source_path):
        source_name = os.path.join(source_path, name)
        if os.path.isdir(source_name):
            print("Skip dir:", source_name)
            continue
        if not source_name.endswith(".jpg"):
            print("Skip unknown file:", source_name)
            continue

        #print("Processing", source_name)
        img = PIL.Image.open(source_name)
        boxes, score = mtcnn.detect(img)
        if boxes is not None:
            if random.random() >= test_split:
                target_name = os.path.join(target_train_path, name)
            else:
                target_name = os.path.join(target_test_path, name)
            face = img.crop(boxes[0])
            face.save(target_name)


def main(args):
    clear_progress()

    with open(os.path.join("/data/input", args.input), "rt", encoding="utf-8") as fp:
        input_params = json.load(fp)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    if "name" not in input_params:
        save_progress({
            "status": "done",
        })
        save_result({
            "name": "dataset01",
            "count": 10
        }, args.output)
        return

    dataset_name = input_params["name"]
    target_path = os.path.join("/dataset", dataset_name)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    count = input_params.get("count", 10)
    test_split = input_params.get("test_split", 0.1)
    if count > 1000:
        save_progress({
            "status": "done",
        })
        save_result({
            "success": False,
            "message": "Count cannot be over 1000"
        }, args.output)
        return
    class_list = []
    with open(os.path.join("/vggface2/meta", "identity_meta.csv"), "rt", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        for row in csvreader:
            class_list.append((row[0], row[1], row[3]))
    class_list = class_list[1:]     # skip header
    total_count = len(class_list)
    print("Total class count:", total_count)
    sample_index = random.sample(range(total_count), count)
    samples = [class_list[x] for x in sample_index]

    mtcnn_params = input_params.get("mtcnn", {})
    mtcnn = MTCNN(
        **mtcnn_params,
        device=device
    )

    current_count = 0
    for sample in samples:
        class_id, name, flag = sample
        print(class_id, name)
        save_progress({
            "status": "running",
            "count": count,
            "current_count": current_count,
            "current_sample": [class_id, name]
        })
        input_path = os.path.join("/vggface2/data", "train" if flag == "1" else "test", class_id)
        train_path = os.path.join(target_path, "train", name)
        test_path = os.path.join(target_path, "test", name)
        detect_face(mtcnn, input_path, train_path, test_path, test_split)
        current_count += 1
    save_progress({
        "status": "done",
        "count": count,
        "current_count": current_count,
        "current_sample": [class_id, name]
    })
    save_result({
        "samples": samples
    }, args.output)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
