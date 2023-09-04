from mtcnn import MTCNN
import cv2
import os
import argparse
import json

def main(args):
    image_path = None
    if os.path.exists(args.input):
        image_path = args.input
    elif os.path.exists(os.path.join("data", args.input)):
        image_path = os.path.join("data", args.input)
    if image_path is not None:
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        result = detector.detect_faces(img)
    else:
        print("File not found:", args.input)
        result = []

    with open(os.path.join("output", args.output), "wt") as fp:
        json.dump(result, fp)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="test.jpg")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()

if __name__ == "__main__":
    print("Run MTCNN main on:", os.path.abspath("."))
    main(parse_arguments())