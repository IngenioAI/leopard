import argparse
import os
import json

import shutil
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

app = FastAPI()

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
    if mode == "list_model":
        list_model = os.listdir("/model")
        model_info_list = []
        for model_name in list_model:
            model_dir = os.path.join("/model", model_name)
            model_info_path = os.path.join(model_dir, "info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, "rt", encoding="utf-8") as fp:
                    model_info = json.load(fp)
            else:
                model_info = None

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

def web_main(args):
    app.args = args
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=12730)
    return parser.parse_args()


if __name__ == '__main__':
    web_main(parse_arguments())
