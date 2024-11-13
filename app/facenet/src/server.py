'''
    Facenet inference server
'''
import argparse
import os
import json
import random

import urllib.request
from urllib.parse import urlparse
import shutil

import torch
import numpy as np
import PIL

from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn


app = FastAPI()

class FaceNetServer():
    def __init__(self, model_path, label_path=None, freeze=False):
        self.freeze = freeze
        self.model_path = model_path

        if label_path is not None:
            with open(label_path, "rt", encoding="utf-8") as fp:
                self.label = json.load(fp)
            self.num_classes = len(self.label.items())
        else:
            self.label = None
            self.num_classes = 10

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = InceptionResnetV1(
            pretrained='vggface2' if not os.path.exists(self.model_path) else None,
            classify=True,
            num_classes=self.num_classes,
            dropout_prob=0.2
        )

        if self.freeze:
            # freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

            # unfreeze last layer
            last_layer = list(self.model.modules())[-1]
            print("Freeze all but below layer:")
            print(last_layer)
            print(" ")
            last_layer.weight.requires_grad = True

        self.model.to(self.device)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            print("Trained model is NOT found:", self.model_path)

        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=40,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
            device=self.device
        )

    def face_recognize(self, image_path, is_test=False):
        trans = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        img = PIL.Image.open(image_path)
        if is_test:
            boxes = np.array([[0, 0, img.width, img.height]])
            probs = np.array([[1.0]])
        else:
            boxes, probs, _ = self.mtcnn.detect(img, landmarks=True)

        faces = []
        if boxes is not None:
            for box in boxes:
                margin = 0.6
                b_w = box[2] - box[0]
                b_h = box[3] - box[1]
                box[0] = max(box[0] - b_w * margin, 0)
                box[1] = max(box[1] - b_h * margin, 0)
                box[2] = min(box[2] + b_w * margin, img.width)
                box[3] = min(box[3] + b_h * margin, img.height)
                print(box)
                face = img.crop(box)
                faces.append(trans(face))

            faces = torch.stack(faces, dim=0).to(self.device)
            self.model.eval()
            outputs = self.model(faces)
            softmax_outputs = []
            for output in outputs.tolist():
                output_exp = np.exp(output)
                output = output_exp / sum(output_exp)
                softmax_outputs.append(output.tolist())
            outputs = softmax_outputs
            return boxes.tolist(), probs.tolist(), outputs
        return [], [], []


@app.get("/api/ping")
async def server_ping():
    return JSONResponse({
        "success": True
    })

@app.post("/api/run")
async def run_app(req: Request):
    params = await req.json()
    print(params)
    mode = params.get("mode", "inference")
    is_test = False
    if mode == "inference":
        if 'image_url' in params:
            image_url = params['image_url']
            image_url_info = urlparse(image_url)
            image_path = os.path.join("/data/input", os.path.basename(image_url_info.path))
            urllib.request.urlretrieve(image_url, image_path)
        elif 'image_path' in params:
            image_path = os.path.join("/data/input", params['image_path'])
        elif 'test_data_path' in params:
            image_path = os.path.join("/dataset", params['test_data_path'])
            is_test = True
        boxes, probs, preds = app.model.face_recognize(image_path, is_test)
        return JSONResponse({
            "boxes": boxes,
            "face_confidence": probs,
            "predictions": preds,
            "max_index": [p.index(max(p)) for p in preds],
            "label": app.model.label
        })
    elif mode == "load":
        model_path = os.path.join("/model", params["model_name"], "model.pth")
        label_path = os.path.join("/model", params["model_name"], "class_to_idx.json")
        if os.path.exists(model_path):
            app.model = FaceNetServer(model_path, label_path = label_path)
            return JSONResponse({
                "success": True
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "Model not found"
            })
    elif mode == "list_model":
        list_model = os.listdir("/model")
        model_info_list = []
        for model_name in list_model:
            model_dir = os.path.join("/model", model_name)
            model_info_path = os.path.join(model_dir, "info.json")
            with open(model_info_path, "rt", encoding="utf-8") as fp:
                model_info = json.load(fp)
                model_info_list.append({
                    "name": model_name,
                    "info": model_info
                })

        return JSONResponse({
            "success": True,
            "list_model": model_info_list
        })
    elif mode == "remove_model":
        model_name = params["model_name"]
        model_path = os.path.join("/model", model_name)
        try:
            shutil.rmtree(model_path)
        except Exception as e:
            print(e)
    elif mode == "list_dataset":
        list_dataset = os.listdir("/dataset")
        return JSONResponse({
            "success": True,
            "list_dataset": list_dataset
        })
    elif mode == "list_random_test_data":
        dataset_name = params["dataset"]
        model_name = params["model"]
        class_id = params["class_id"]
        count = params["count"]

        label_info_path = os.path.join("/model", model_name, "class_to_idx.json")
        with open(label_info_path, "rt", encoding="utf-8") as fp:
            label_info = json.load(fp)

        root = os.path.join("/dataset", dataset_name, "test")
        prefix_len = len("/dataset/")
        file_list = []
        for r, d, f in os.walk(root):
            for file in f:
                file_list.append(os.path.join(r[prefix_len:], file))
        print(len(file_list))
        selected_list = random.sample(file_list, count)
        label_list = []
        for sample_path in selected_list:
            last_dir_name = sample_path.split("/")[-2]
            label_list.append(last_dir_name)
        return JSONResponse({
            "success": True,
            "random_test_data_list": selected_list,
            "labels": label_list
        })



@app.get("/api/data/{data_path:path}")
async def get_data(data_path: str):
    data_file_path = os.path.join("/dataset", data_path)
    if os.path.exists(data_file_path) and not os.path.isdir(data_file_path):
        if os.access(data_file_path, os.R_OK):
            return FileResponse(data_file_path)
        raise HTTPException(status_code=503, detail="File access not allowed")
    raise HTTPException(status_code=404, detail="File not found")

def web_main(args):
    app.args = args
    if os.path.exists(args.model_path):
        app.model = FaceNetServer(args.model_path, label_path=args.label_path)
    else:
        app.model = None
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--label_path", type=str, default=None)
    parser.add_argument('--port', type=int, default=12720)
    return parser.parse_args()


if __name__ == '__main__':
    web_main(parse_arguments())
