from typing import Union, List
import os
import json
import argparse
import shutil

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, Response, Form, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.status import HTTP_302_FOUND, HTTP_303_SEE_OTHER

import storage_util
import data_store
from html_util import process_include_html
import upload_util
from fastapi_util import JSONResponseHandler
from app.manager import AppManager
from exec import exec_router
import sysinfo
from session import session_router
from image import image_router

tags_metadata = [
    {"name": "UI", "description": "UI template service"},
    {"name": "Session", "description": "Session management"},
    {"name": "Image", "description": "Docker image management"},
    {"name": "Exec", "description": "Experiment execution manager"}
]

app = FastAPI(openapi_tags=tags_metadata)

app.include_router(session_router)
app.include_router(image_router)
app.include_router(exec_router)

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


@app.get("/", tags=["UI"])
async def get_index_file():
    return RedirectResponse(url="/ui/index.html", status_code=HTTP_303_SEE_OTHER)


@app.get("/api/storage", tags=["Storage"])
async def get_storage_list():
    userStorageList = storage_util.get_storage_info("user")
    return JSONResponseHandler([{"id": x["id"], "name": x["name"]} for x in userStorageList])


@app.get("/api/storage/{storage_id}", tags=["Storage"])
async def get_storage_file_list(storage_id: str, page: int = 0, count: int = 0):
    file_list = storage_util.get_file_list(storage_id)
    total_count = len(file_list)
    if count != 0:
        file_list = file_list[page * count: (page + 1) * count]
    file_info_list = storage_util.get_file_info(storage_id, ".", file_list)
    return JSONResponseHandler({
        'id': storage_id,
        'file_path': '.',
        'page': page,
        'count': count,
        'total_count': total_count,
        'items': file_info_list
    })


@app.get("/api/storage/{storage_id}/{file_path:path}", tags=["Storage"])
async def get_storage_file_list_with_path(storage_id: str, file_path: str, page: int = 0, count: int = 0):
    file_list = storage_util.get_file_list(storage_id, file_path)
    total_count = len(file_list)
    if count != 0:
        file_list = file_list[page * count: (page + 1) * count]
    file_info_list = storage_util.get_file_info(storage_id, file_path, file_list)
    return JSONResponseHandler({
        'id': storage_id,
        'file_path': file_path,
        'page': page,
        'count': count,
        'total_count': total_count,
        'items': file_info_list
    })


@app.put("/api/storage/{storage_id}/{file_path:path}", tags=["Storage"])
async def create_storage_directory(storage_id: str, file_path: str):
    storage_file_path = storage_util.get_storage_file_path(storage_id, file_path)
    if os.path.exists(storage_file_path):
        raise HTTPException(status_code=403, detail="File already exist")
    os.mkdir(storage_file_path)
    return JSONResponseHandler({
        'success': True,
        'dir': storage_file_path
    })


@app.get("/api/storage_file/{storage_id}/{file_path:path}", tags=["Storage"])
async def get_storage_file(storage_id: str, file_path: str):
    storage_file_path = storage_util.get_storage_file_path(storage_id, file_path)
    if os.path.exists(storage_file_path) and not os.path.isdir(storage_file_path):
        if os.access(storage_file_path, os.R_OK):
            return FileResponse(storage_file_path)
        else:
            raise HTTPException(status_code=503, detail="File access not allowed")
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/api/storage_file/{storage_id}/{file_path:path}", tags=["Storage"])
async def post_storage_file(storage_id: str, file_path: str, req: Request):
    storage_file_path = storage_util.get_storage_file_path(storage_id, file_path)
    if not os.path.exists(storage_file_path):
        raise HTTPException(status_code=404, detail="Path not found")
    file_list, metadata = await upload_util.handle_upload(req, storage_file_path)
    return JSONResponseHandler({
        'success': True,
        'files': file_list
    })


@app.put("/api/storage_file/{storage_id}/{file_path:path}", tags=["Storage"])
async def save_storage_file(storage_id: str, file_path: str, req: Request):
    storage_file_path = storage_util.get_storage_file_path(storage_id, file_path)
    contents = await req.body()
    with open(storage_file_path, "wb") as fp:
        fp.write(contents)

    return JSONResponseHandler({
        'success': True
    })

@app.post("/api/upload_item", tags=["Storage"])
async def upload_item(req: Request):
    target_dir, upload_id = upload_util.get_upload_item_dir()
    storage_util.ensure_path(target_dir)
    file_list, metadata = await upload_util.handle_upload(req, target_dir)
    unzip_file_list = []
    if "unzip" in metadata and metadata["unzip"]:
        for file in file_list:
            unzip_file_list += upload_util.get_compressed_filelist(os.path.join(target_dir, file))
    return JSONResponseHandler({
        "success": True,
        "files": file_list,
        "unzip_files": unzip_file_list,
        "upload_id": upload_id
    })

@app.delete("/api/upload_item/{upload_id}", tags=["Storage"])
async def delete_upload_item(upload_id: str):
    target_dir = os.path.join("storage", "upload", upload_id)
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
        except:
            pass
    return JSONResponse({
        "success": True
    })


@app.delete("/api/storage/{storage_id}/{file_path:path}", tags=["Storage"])
async def delete_storage_file(storage_id: str, file_path: str, req: Request):
    storage_file_path = storage_util.get_storage_file_path(storage_id, file_path)
    if os.path.exists(storage_file_path):
        try:
            if os.path.isdir(storage_file_path):
                os.rmdir(storage_file_path)
            else:
                os.unlink(storage_file_path)
            return JSONResponseHandler({
                'success': True
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.get("/api/dataset", tags=["Dataset"])
async def get_dataset_list():
    return JSONResponseHandler(data_store.manager.get_data_list("dataset"))

@app.post("/api/dataset", tags=["Dataset"])
async def post_dataset_list(req: Request):
    dataset_list = await req.json()
    res = data_store.manager.save_data_list("dataset", dataset_list)
    return JSONResponseHandler({
        "success": res
    })

@app.post("/api/dataset/{name}", tags=["Dataset"])
async def add_dataset(req: Request):
    dataset = await req.json()
    res = data_store.manager.add_data_to_list("dataset", dataset)
    return JSONResponseHandler({
        "success": res
    })

@app.delete("/api/dataset/{name}", tags=["Dataset"])
async def delete_dataset(name: str):
    data_store.manager.remove_data_from_list("dataset", "name", name)
    return JSONResponseHandler({
        "success": True
    })

@app.get("/api/model", tags=["Model"])
async def get_model_list():
    return JSONResponseHandler(data_store.manager.get_data_list("model"))

@app.post("/api/model", tags=["Model"])
async def post_model_list(req: Request):
    model_list = await req.json()
    res = data_store.manager.save_data_list("model", model_list)
    return JSONResponseHandler({
        "success": res
    })

@app.post("/api/model/{name}", tags=["Model"])
async def add_model(req: Request):
    model = await req.json()
    res = data_store.manager.add_data_to_list("model", model)
    return JSONResponseHandler({
        "success": res
    })

@app.delete("/api/model/{name}", tags=["Model"])
async def delete_model(name: str):
    data_store.manager.remove_data_from_list("model", "name", name)
    return JSONResponseHandler({
        "success": True
    })

@app.get("/api/app_list", tags=["App"])
async def get_app_list():
    return JSONResponseHandler(app.app_manager.app_info)

@app.post("/api/app/{module_id}", tags=["App"])
async def run_app(module_id: str, req: Request):
    params = await req.json()
    res = app.app_manager.run(module_id, params)
    return JSONResponseHandler(res)

@app.get("/api/sys_info", tags=["System"])
async def get_sys_info():
    return JSONResponseHandler(app.sys_info.get_system_info())

'''
@app.post("/api/create_session/{name}", tags=["Session"])
async def create_session(name: str, response: Response):
    session = uuid4()
    data = SessionData(username=name)
    await session_manager.backend.create(session, data)
    session_manager.cookie.attach_to_response(response, session)
    return {
        "success": True,
        "username": name,
        "session": str(session)
    }    # do not use JSONResponse, cookie cannot be attached to JSONResponse

@app.get("/api/session", dependencies=[Depends(session_manager.cookie)], tags=["Session"])
async def get_session(session_data: SessionData = Depends(session_manager.verifier)):
    return JSONResponseHandler({
        "success": True,
        "username": session_data.username
    })

@app.delete("/api/session", tags=["Session"])
async def delete_session(response: Response, session_id: UUID = Depends(session_manager.cookie)):
    await session_manager.backend.delete(session_id)
    session_manager.cookie.delete_from_response(response)
    return {
        "success": True
    }
'''

app.mount("/", StaticFiles(directory="webroot"), name="static")


def web_main(args):
    app.args = args
    app.app_manager = AppManager()
    app.app_manager.start()
    app.sys_info = sysinfo.SystemInfo()
    app.sys_info.start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

    print("Cleanup app docker")
    app.app_manager.stop()
    app.sys_info.stop()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=12700)
    return parser.parse_args()


if __name__ == '__main__':
    web_main(parse_arguments())
