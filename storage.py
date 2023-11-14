import os
import shutil

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

import storage_util
import upload_util
from fastapi_util import JSONResponseHandler

storage_router = APIRouter(prefix="/api/storage", tags=["Storage"])

@storage_router.get("/list")
async def get_storage_list():
    userStorageList = storage_util.get_storage_info("user")
    return JSONResponseHandler([{"id": x["id"], "name": x["name"]} for x in userStorageList])


@storage_router.get("/list/{storage_id}")
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


@storage_router.get("/list/{storage_id}/{file_path:path}")
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


@storage_router.put("/dir/{storage_id}/{file_path:path}")
async def create_storage_directory(storage_id: str, file_path: str):
    storage_file_path = storage_util.get_storage_file_path(storage_id, file_path)
    if os.path.exists(storage_file_path):
        raise HTTPException(status_code=403, detail="File already exist")
    os.mkdir(storage_file_path)
    return JSONResponseHandler({
        'success': True,
        'dir': storage_file_path
    })


@storage_router.get("/file/{storage_id}/{file_path:path}")
async def get_storage_file(storage_id: str, file_path: str):
    storage_file_path = storage_util.get_storage_file_path(storage_id, file_path)
    if os.path.exists(storage_file_path) and not os.path.isdir(storage_file_path):
        if os.access(storage_file_path, os.R_OK):
            return FileResponse(storage_file_path)
        else:
            raise HTTPException(status_code=503, detail="File access not allowed")
    raise HTTPException(status_code=404, detail="File not found")


@storage_router.post("/file/{storage_id}/{file_path:path}")
async def post_storage_file(storage_id: str, file_path: str, req: Request):
    storage_file_path = storage_util.get_storage_file_path(storage_id, file_path)
    if not os.path.exists(storage_file_path):
        raise HTTPException(status_code=404, detail="Path not found")
    file_list, metadata = await upload_util.handle_upload(req, storage_file_path)
    return JSONResponseHandler({
        'success': True,
        'files': file_list
    })


@storage_router.put("/file/{storage_id}/{file_path:path}")
async def save_storage_file(storage_id: str, file_path: str, req: Request):
    storage_file_path = storage_util.get_storage_file_path(storage_id, file_path)
    contents = await req.body()
    with open(storage_file_path, "wb") as fp:
        fp.write(contents)

    return JSONResponseHandler({
        'success': True
    })

@storage_router.delete("/item/{storage_id}/{file_path:path}", tags=["Storage"])
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

@storage_router.post("/upload_item")
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

@storage_router.delete("/upload_item/{upload_id}")
async def delete_upload_item(upload_id: str):
    target_dir = os.path.join("storage", "upload", upload_id)
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
        except:
            pass
    return JSONResponseHandler({
        "success": True
    })
