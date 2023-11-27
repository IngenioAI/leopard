import os
import json
import argparse

import uvicorn
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException
from starlette.status import HTTP_303_SEE_OTHER

import data_store
from html_util import process_include_html
from fastapi_util import JSONResponseHandler
from app.manager import AppManager
from exec import exec_router, exec_manager
import sysinfo
from session import session_router, session_manager
from image import image_router
from storage import storage_router

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
    page_path = "ui/page/%s" % file_path
    if os.path.exists(page_path):
        with open(page_path, "rt", encoding="UTF-8") as fp:
            content = fp.read()
        content = process_include_html(content, {'query_param': json.dumps(query_param)}, {"file_path": file_path})
    else:
        with open("ui/page/error.html", "rt", encoding="UTF-8") as fp:
            content = fp.read()
        content = process_include_html(content, {'error_message': "Page not found: %s" % file_path}, {"file_path": file_path})
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
    page_path = "ui/page/%s" % file_path
    if os.path.exists(page_path):
        with open(page_path, "rt", encoding="UTF-8") as fp:
            content = fp.read()
        content = process_include_html(content, {'query_param': json.dumps(query_param)}, {"file_path": file_path})
    else:
        with open("ui/page/error.html", "rt", encoding="UTF-8") as fp:
            content = fp.read()
        content = process_include_html(content, {'error_message': "Page not found: %s" % file_path}, {"file_path": file_path})
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

@app.delete("/api/model/item/{name}", tags=["Model"])
async def delete_model(name: str):
    data_store.manager.remove_data_from_list("model", "name", name)
    return JSONResponseHandler({
        "success": True
    })

@app.get("/api/app/list", tags=["App"])
async def get_app_list():
    return JSONResponseHandler(app.app_manager.app_info)

@app.post("/api/app/run/{module_id}", tags=["App"])
async def run_app(module_id: str, req: Request):
    params = await req.json()
    res = app.app_manager.run(module_id, params)
    return JSONResponseHandler(res)

@app.get("/api/system/info", tags=["System"])
async def get_sys_info():
    return JSONResponseHandler(app.sys_info.get_system_info())

app.mount("/", StaticFiles(directory="webroot"), name="static")


def web_main(args):
    app.args = args
    app.app_manager = AppManager()
    app.app_manager.start()
    app.sys_info = sysinfo.SystemInfo()
    app.sys_info.start()
    exec_manager.start()
    uvicorn.run(app, host="127.0.0.1", port=args.port)

    print("Cleanup app docker")
    app.app_manager.stop()
    app.sys_info.stop()
    exec_manager.stop()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=12700)
    return parser.parse_args()


if __name__ == '__main__':
    web_main(parse_arguments())
