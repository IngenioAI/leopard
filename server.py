from typing import Union, List
import os
import json
import argparse
import shutil

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, Response, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.status import HTTP_302_FOUND, HTTP_303_SEE_OTHER

import storage_util
import dataset_util
from docker_runner import DockerRunner
from html_util import process_include_html
import upload_util
from app.manager import AppManager


tags_metadata = [
    {"name": "File", "description": "Related to static file serving"}
]

app = FastAPI(openapi_tags=tags_metadata)


def JSONResponseHandler(data):
    try:
        return JSONResponse(data)
    except TypeError as e:
        print(e)
        print(data)
        return {}


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


@app.get("/", tags=["File"])
async def get_index_file():
    return RedirectResponse(url="/ui/index.html", status_code=HTTP_303_SEE_OTHER)


class CreateImageItem(BaseModel):
    name: str
    baseImage: str
    update: Union[bool, None] = None
    aptInstall: Union[str, None] = None
    pipInstall: Union[str, None] = None
    additionalCommand: Union[str, None] = None


@app.post("/api/image_create", tags=["Image"])
async def create_image(data: CreateImageItem):
    ret = app.docker_runner.create_image(data.name, data.baseImage, data.update, data.aptInstall, data.pipInstall,
                                         data.additionalCommand)
    return JSONResponseHandler({
        'success': ret
    })


@app.get("/api/image_create/{name:path}", tags=["Image"])
async def get_image_creation_info(name: str):
    info = app.docker_runner.get_create_image_info(name)
    return JSONResponseHandler(info)


@app.delete("/api/image_create/{name:path}", tags=["Image"])
async def remove_image_creation_info(name: str):
    app.docker_runner.remove_create_image_info(name)
    return JSONResponse({
        'success': True
    })


@app.get("/api/image/list", tags=["Image"])
async def get_image_list():
    return JSONResponseHandler(app.docker_runner.list_images())


@app.delete("/api/image/{name:path}", tags=["Image"])
async def delete_image(name: str):
    app.docker_runner.remove_image(name)
    return JSONResponse({
        "success": True
    })


@app.get("/api/exec_list", tags=["Exec"])
async def get_exec_list():
    return JSONResponseHandler(app.docker_runner.list_execs())

class ExecutionItem(BaseModel):
    id: str
    srcPath: str
    command: str
    imageTag: str
    inputPath: Union[str, None] = None
    outputPath: Union[str, None] = None
    uploadId: Union[str, None] = None


@app.post("/api/exec", tags=["Exec"])
async def create_execution(data: ExecutionItem):
    if data.uploadId is not None:
        source_path = upload_util.process_upload_item(data.uploadId, data.id, data.srcPath)
    else:
        source_path = data.srcPath

    containerId = app.docker_runner.exec_command(source_path, data.command, data.imageTag, data.inputPath,
                                                data.outputPath)
    print('execId(containerId):', containerId)
    return JSONResponseHandler({
        'exec_id': containerId
    })


@app.get("/api/exec/{exec_id}", tags=["Exec"])
async def get_execution_info(exec_id: str):
    info = app.docker_runner.exec_inspect(exec_id)
    return JSONResponseHandler(info)


@app.get("/api/exec/logs/{exec_id}", tags=["Exec"])
async def get_execution_logs(exec_id: str):
    logs = app.docker_runner.exec_logs(exec_id)
    return JSONResponseHandler({
        'lines': logs
    })


@app.delete("/api/exec/{exec_id}", tags=["Exec"])
async def remove_execution_info(exec_id: str):
    app.docker_runner.exec_remove(exec_id)
    return JSONResponse({
        'success': True
    })


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
    return JSONResponseHandler(dataset_util.get_dataset_info())

@app.post("/api/dataset", tags=["Dataset"])
async def post_dataset_list(req: Request):
    info = await req.json()
    return JSONResponseHandler({
        "success": dataset_util.save_dataset_info(info)
    })

@app.get("/api/app_list", tags=["App"])
async def get_app_list():
    return JSONResponseHandler(app.app_manager.app_info)

@app.post("/api/app/{module_id}", tags=["App"])
async def run_app(module_id: str, req: Request):
    params = await req.json()
    res = app.app_manager.run(module_id, params)
    return JSONResponseHandler(res)


app.mount("/", StaticFiles(directory="webroot"), name="static")


def web_main(args):
    app.args = args
    app.docker_runner = DockerRunner()
    app.app_manager = AppManager()
    app.app_manager.start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

    print("Cleanup app docker")
    app.app_manager.stop()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=12700)
    return parser.parse_args()


if __name__ == '__main__':
    web_main(parse_arguments())
