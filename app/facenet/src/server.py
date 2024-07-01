'''
    Facenet as server
'''
import argparse
import os
import torch
import PIL
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn


app = FastAPI()

class FaceNetServer():
    def __init__(self, model_path):
        self.num_classes = 10
        self.freeze = False
        self.model_path = model_path

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

        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )


    def face_recognize(self, image_path):
        trans = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        img = PIL.Image.open(image_path)
        boxes, probs, _ = self.mtcnn.detect(img, landmarks=True)
        faces = []
        if boxes is not None:
            for box in boxes:
                face = img.crop(box)
                faces.append(trans(face))

            faces = torch.stack(faces, dim=0).to(self.device)
            self.model.eval()
            outputs = self.model(faces)
        else:
            outputs = []
        return boxes.tolist(), probs.tolist(), outputs.tolist()

@app.get("/api/ping")
async def server_ping():
    return JSONResponse({
        "success": True
    })

@app.post("/api/run")
async def run_app(req: Request):
    params = await req.json()
    print(params)
    image_path = os.path.join("/data/input", params['image_path'])
    boxes, probs, preds = app.model.face_recognize(image_path)
    return JSONResponse({
        "boxes": boxes,
        "probabilities": probs,
        "predictions": preds,
        "max_index": [p.index(max(p)) for p in preds]
    })

def web_main(args):
    app.args = args
    app.model = FaceNetServer(args.model_path)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="inference")        # inference / train / unlearning
    parser.add_argument("--model_path", type=str, default="model/VGGFace2_10.pth")
    parser.add_argument('--port', type=int, default=12720)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    web_main(args)
