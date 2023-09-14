from typing import Union, List
import os
import argparse

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, Response, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.status import HTTP_302_FOUND, HTTP_303_SEE_OTHER

import storage_util
from docker_runner import DockerRunner
from html_util import process_include_html
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


async def handle_upload(req: Request, target_dir, with_content=False):
    data = await req.form()
    file_list = []
    meta_data = {}
    content_list = []
    for filename in data:
        if filename[0] == '/':
            meta_data[filename[1:]] = data[filename]
        else:
            content = await data[filename].read()
            print('FILE', filename)
            file_path = os.path.join(target_dir, filename)
            with open(file_path, "wb") as fp:
                fp.write(content)
            file_list.append(filename)
            if with_content:
                content_list.append(content)

    if with_content:
        return file_list, content_list, meta_data
    return file_list, meta_data


@app.get("/ui/{file_path:path}", tags=["UI"])
async def get_ui_page(file_path: str):
    page_path = "ui/template/%s" % file_path
    if os.path.exists(page_path):
        with open(page_path, "rt", encoding="UTF-8") as fp:
            content = fp.read()
        content = process_include_html(content)
    else:
        with open("ui/template/error.html", "rt", encoding="UTF-8") as fp:
            content = fp.read()
        content = process_include_html(content, {'error_message': "Page not found: %s" % file_path})
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


@app.post("/api/image/create", tags=["Image"])
async def create_image(data: CreateImageItem):
    ret = app.docker_runner.create_image(data.name, data.baseImage, data.update, data.aptInstall, data.pipInstall,
                                         data.additionalCommand)
    return JSONResponseHandler({
        'success': ret
    })


@app.get("/api/image/create/{name}", tags=["Image"])
async def get_image_creation_info(name: str):
    info = app.docker_runner.get_create_image_info(name)
    return JSONResponseHandler(info)


@app.delete("/api/image/create/{name}", tags=["Image"])
async def remove_image_creation_info(name: str):
    app.docker_runner.remove_create_image_info(name)
    return JSONResponse({
        'success': True
    })


@app.get("/api/image/list", tags=["Image"])
async def get_image_list():
    return JSONResponseHandler(app.docker_runner.list_images())


class ExecutionItem(BaseModel):
    srcPath: str
    mainSrc: str
    imageTag: str
    inputPath: Union[str, None] = None
    outputPath: Union[str, None] = None


@app.post("/api/exec", tags=["Exec"])
async def create_execution(data: ExecutionItem):
    containerId = app.docker_runner.exec_python(data.srcPath, data.mainSrc, data.imageTag, data.inputPath,
                                                data.outputPath)
    print('execId(containerId):', containerId)
    return JSONResponseHandler({
        'exec_id': containerId
    })


@app.get("/api/exec/{exec_id}", tags=["Exec"])
async def get_execution_info(exec_id: str):
    info = app.docker_runner.exec_inspect(exec_id)
    print('inspect', info)
    return JSONResponseHandler(info)


@app.get("/api/exec/logs/{exec_id}", tags=["Exec"])
async def get_execution_logs(exec_id: str):
    logs = app.docker_runner.exec_logs(exec_id)
    print('logs', logs)
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
    return JSONResponseHandler([
        {
            'name': 'default',
            'id': '1'
        }
    ])


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
        return FileResponse(storage_file_path)
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/api/storage_file/{storage_id}/{file_path:path}", tags=["Storage"])
async def post_storage_file(storage_id: str, file_path: str, req: Request):
    storage_file_path = storage_util.get_storage_file_path(storage_id, file_path)
    if not os.path.exists(storage_file_path):
        raise HTTPException(status_code=404, detail="Path not found")
    file_list, metadata = await handle_upload(req, storage_file_path)
    return JSONResponseHandler({
        'success': True,
        'files': file_list
    })


@app.put("/api/storage_file/{storage_id}/{file_path:path}", tags=["Storage"])
async def save_storage_file(storage_id: str, file_path: str, req: Request):
    storage_file_path = storage_util.get_storage_file_path(storage_id, file_path)
    contents = await req.body()
    if os.path.exists(storage_file_path):
        raise HTTPException(status_code=403, detail="File already exist")
    with open(storage_file_path, "wb") as fp:
        fp.write(contents)

    return JSONResponseHandler({
        'success': True
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


@app.post("/api/app/{module_id}", tags=["App"])
async def call_app(module_id: str, req: Request):
    params = await req.json()
    res = app.app_manager.call(module_id, params)
    return JSONResponseHandler(res)


app.mount("/", StaticFiles(directory="webroot"), name="static")


def web_main(args):
    app.args = args
    app.docker_runner = DockerRunner()
    app.app_manager = AppManager()
    app.app_manager.run()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

    print("Cleanup app docker")
    app.app_manager.stop()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=12700)
    return parser.parse_args()


if __name__ == '__main__':
    web_main(parse_arguments())
