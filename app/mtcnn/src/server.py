'''
    MTCNN module as server

'''
import argparse
import copy
import json
import os
from urllib import request

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from mtcnn import MTCNN
import tensorflow as tf

app = FastAPI()


def average_filter(mat, channels_first=True, alpha=False):
    # print("before average", mat)
    if channels_first:
        ch = 0
        new_mat = np.ones_like(mat, dtype=mat.dtype)
        for i in range(mat.shape[ch]):
            if alpha and i == 3:
                new_mat[i, :, :] = mat[i, :, :]
            else:
                new_mat[i, :, :] = np.ones_like(mat[i, :, :], dtype=mat.dtype) * np.mean(mat[i, :, :])
    else:
        ch = 2
        new_mat = np.ones_like(mat, dtype=mat.dtype)
        for i in range(mat.shape[ch]):
            if alpha and i == 3:
                new_mat[:, :, i] = mat[:, :, i]
            else:
                new_mat[:, :, i] = np.ones_like(mat[:, :, i], dtype=mat.dtype) * np.mean(mat[:, :, i])

    # print("after average", new_mat)
    return new_mat


def anonymize_faces(image_path, boxes, channels_first=False, alpha=False):
    mat = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    print("boxes", boxes)
    new_mat = copy.deepcopy(mat)

    for box in boxes:
        x, y, width, height = box['box']
        l = x
        r = x + width
        t = y
        b = y + height
        if channels_first:
            blurred_face = average_filter(mat[:, t:b, l:r], channels_first=channels_first, alpha=alpha)
            new_mat[:, t:b, l:r] = blurred_face
        else:
            blurred_face = average_filter(mat[t:b, l:r, :], channels_first=channels_first, alpha=alpha)
            new_mat[t:b, l:r, :] = blurred_face

    new_img = cv2.cvtColor(new_mat, cv2.COLOR_RGB2BGR)
    new_img_name = "blurred_img.png"
    cv2.imwrite(os.path.join('/data/input', new_img_name), new_img)
    cv2.imwrite(os.path.join('/data/output', new_img_name), new_img)

    return new_img_name


def detect_faces(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    return detector.detect_faces(img)

@app.get("/api/ping")
async def server_ping():
    return JSONResponse({
        "success": True
    })

@app.post("/api/run", tags=["App"])
async def run_app(req: Request):
    params = await req.json()
    print(params)
    image_path = params['image_path']
    if not os.path.exists(image_path) and os.path.exists(os.path.join("/data/input", image_path)):
        image_path = os.path.join("/data/input", image_path)

    mode = params['mode']
    if image_path:
        boxes = detect_faces(image_path)
        # FIXME 추후 response 객채 정의 하여 status 값 및 error message 전달 필요.
        if len(boxes) <= 0:
            return JSONResponse({})
        if mode == 'detect':
            return JSONResponse(boxes)
        if mode == 'anonymize':
            result = {}
            new_img_name = anonymize_faces(image_path, boxes)
            result["image_path"] = new_img_name
            return JSONResponse(result)

        return JSONResponse({})
    raise HTTPException(status_code=404)


def web_main(args):
    app.args = args
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=12710)
    return parser.parse_args()


if __name__ == '__main__':
    try:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

    web_main(parse_arguments())
