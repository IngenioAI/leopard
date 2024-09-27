import os
import json
import argparse

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException
from starlette.status import HTTP_303_SEE_OTHER

import data_store
from html_util import process_lp_html
from fastapi_util import JSONResponseHandler
from app.manager import app_manager
from exec import exec_router, exec_manager
from sysinfo import sys_info
from session import session_router, session_manager
from image import image_router
from storage import storage_router
import tensorboard

tags_metadata = [
    {"name": "UI", "description": "UI template service"},
    {"name": "Session", "description": "Session management"},
    {"name": "Image", "description": "Docker image management"},
    {"name": "Exec", "description": "Experiment execution manager"},
    {"name": "Storage", "description": "Storage file manager"}
]

app = FastAPI(openapi_tags=tags_metadata)

app.include_router(session_router)
app.include_router(image_router)
app.include_router(exec_router)
app.include_router(storage_router)

@app.get("/ui/{file_path:path}", tags=["UI"])
async def get_ui_page(file_path: str, req: Request):
    query_param = dict(req.query_params)
    page_path = f'ui/page/{file_path}'
    if os.path.exists(page_path):
        with open(page_path, "rt", encoding="UTF-8") as fp:
            content = fp.read()
        content = process_lp_html(content, {'query_param': json.dumps(query_param)}, {"file_path": file_path})
    else:
        with open("ui/page/error.html", "rt", encoding="UTF-8") as fp:
            content = fp.read()
        content = process_lp_html(content, {'error_message': f'Page not found: {file_path}'}, {"file_path": file_path})
    return HTMLResponse(content, status_code=200)

@app.get("/ui_secure_test/{file_path:path}", tags=["UI"])
async def get_ui_secure_page(file_path: str, req: Request):
    try:
        session_manager.cookie(req)
        session_data = await session_manager.verifier(req)
    except HTTPException as e:
        print(e, e.detail, type(e))
        return RedirectResponse(url="/ui/login.html", status_code=HTTP_303_SEE_OTHER)
    print("Secure page:", session_data)
    query_param = dict(req.query_params)
    page_path = f'ui/page/{file_path}'
    if os.path.exists(page_path):
        with open(page_path, "rt", encoding="UTF-8") as fp:
            content = fp.read()
        content = process_lp_html(content, {'query_param': json.dumps(query_param)}, {"file_path": file_path})
    else:
        with open("ui/page/error.html", "rt", encoding="UTF-8") as fp:
            content = fp.read()
        content = process_lp_html(content, {'error_message': f'Page not found: {file_path}'}, {"file_path": file_path})
    return HTMLResponse(content, status_code=200)


@app.get("/", tags=["UI"])
async def get_index_file():
    return RedirectResponse(url="/ui/index.html", status_code=HTTP_303_SEE_OTHER)


@app.get("/api/dataset/list", tags=["Dataset"])
async def get_dataset_list():
    return JSONResponseHandler(data_store.manager.get_data_list("dataset"))

@app.post("/api/dataset/list", tags=["Dataset"])
async def post_dataset_list(req: Request):
    dataset_list = await req.json()
    res = data_store.manager.save_data_list("dataset", dataset_list)
    return JSONResponseHandler({
        "success": res
    })

@app.post("/api/dataset/item/{name}", tags=["Dataset"])
async def add_dataset(req: Request):
    dataset = await req.json()
    res = data_store.manager.add_data_to_list("dataset", dataset)
    return JSONResponseHandler({
        "success": res
    })

@app.put("/api/dataset/item/{name}", tags=["Dataset"])
async def update_dataset(req: Request):
    dataset = await req.json()
    res = data_store.manager.update_data_in_list("dataset", dataset, "name")
    return JSONResponseHandler({
        "success": res
    })

@app.delete("/api/dataset/item/{name}", tags=["Dataset"])
async def delete_dataset(name: str):
    data_store.manager.remove_data_from_list("dataset", "name", name)
    return JSONResponseHandler({
        "success": True
    })

@app.get("/api/model/list", tags=["Model"])
async def get_model_list():
    return JSONResponseHandler(data_store.manager.get_data_list("model"))

@app.post("/api/model/list", tags=["Model"])
async def post_model_list(req: Request):
    model_list = await req.json()
    res = data_store.manager.save_data_list("model", model_list)
    return JSONResponseHandler({
        "success": res
    })

@app.post("/api/model/item/{name}", tags=["Model"])
async def add_model(req: Request):
    model = await req.json()
    res = data_store.manager.add_data_to_list("model", model)
    return JSONResponseHandler({
        "success": res
    })

@app.put("/api/model/item/{name}", tags=["Model"])
async def update_model(req: Request):
    model = await req.json()
    res = data_store.manager.update_data_in_list("model", model, "name")
    return JSONResponseHandler({
        "success": res
    })

@app.delete("/api/model/item/{name}", tags=["Model"])
async def delete_model(name: str):
    data_store.manager.remove_data_from_list("model", "name", name)
    return JSONResponseHandler({
        "success": True
    })

@app.get("/api/app/list", tags=["App"])
async def get_app_list():
    return JSONResponseHandler(app_manager.app_info)

@app.post("/api/app/run/{module_id}", tags=["App"])
async def run_app(module_id: str, req: Request):
    params = await req.json()
    res = app_manager.run(module_id, params)
    return JSONResponseHandler(res)

@app.get("/api/app/progress/{module_id}", tags=["App"])
async def get_app_progress(module_id: str):
    res = app_manager.get_progress(module_id)
    return JSONResponseHandler(res)

@app.get("/api/app/logs/{module_id}", tags=["App"])
async def get_app_logs(module_id: str):
    logs = app_manager.get_logs(module_id)
    return JSONResponseHandler({
        "success": True,
        "logs": logs
    })

@app.get("/api/app/result/{module_id}", tags=["App"])
async def get_app_result(module_id: str):
    res = app_manager.get_result(module_id)
    return JSONResponseHandler(res)

@app.get("/api/app/stop/{module_id}", tags=["App"])
async def stop_app(module_id: str):
    res = app_manager.stop_app(module_id)
    return JSONResponseHandler(res)

@app.get("/api/app/remove/{module_id}", tags=["App"])
async def remove_app(module_id: str):
    res = app_manager.remove_app(module_id)
    return JSONResponseHandler(res)

@app.get("/api/app/data/{module_id}/{data_path:path}", tags=["App"])
async def get_app_data(module_id: str, data_path: str):
    content, content_type = app_manager.get_data(module_id, data_path)
    if content is not None:
        return Response(content, media_type=content_type)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/system/info", tags=["System"])
async def get_sys_info():
    return JSONResponseHandler(sys_info.get_system_info())

@app.get("/api/tensorboard/start/{exec_id}", tags=["Tensorboard"])
async def get_start_tensorboard(exec_id: str):
    run_path = exec_manager.get_run_path(exec_id)
    default_port = 12760
    success = tensorboard.manager.start(run_path, default_port)
    return {
        "success": success,
        "port": default_port
    }

@app.get("/api/tensorboard/stop", tags=["Tensorboard"])
async def get_stop_tensorboard():
    tensorboard.manager.stop()

app.mount("/", StaticFiles(directory="webroot"), name="static")

def init_app(config=None):
    app_manager.start(config)
    sys_info.start(config)
    exec_manager.start(config)
    data_store.manager.start(config)

def deinit_app():
    app_manager.stop()
    sys_info.stop()
    exec_manager.stop()
    tensorboard.manager.stop()

def web_main(args):
    app.args = args
    config = None
    if args.config is not None:
        if os.path.exists(args.config):
            with open(args.config, "rt", encoding="utf-8") as fp:
                config = json.load(fp)
    init_app(config)
    uvicorn.run(app, host="0.0.0.0", port=args.port if args else 12700)
    deinit_app()



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--port', type=int, default=12700)
    return parser.parse_args()


if __name__ == '__main__':
    web_main(parse_arguments())
