'''
    MTCNN module as server

'''
import os
import argparse

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse

from mtcnn import MTCNN
import cv2
import json


app = FastAPI()

@app.get("/api/run/{file_path:path}")
async def run_app(file_path: str):
    print("run_app:", file_path)
    if os.path.exists(file_path):
        img_path = file_path
    elif os.path.exists(os.path.join("/data/input", file_path)):
        img_path = os.path.join("/data/input", file_path)
    else:
        img_path = None
    if img_path:
        print("Read image:", img_path)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        result = app.mtcnn.detect_faces(img)
        print(result)
        return JSONResponse(result)
    raise HTTPException(status_code=404)

def web_main(args):
    app.args = args
    app.mtcnn = MTCNN()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=12710)
    return parser.parse_args()


if __name__ == '__main__':
    web_main(parse_arguments())
