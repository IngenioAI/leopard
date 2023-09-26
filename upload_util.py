import os
import time
import zipfile
import tarfile
import shutil

from fastapi import Request

import storage_util

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
            file_path = os.path.join(target_dir, filename)
            with open(file_path, "wb") as fp:
                fp.write(content)
            file_list.append(filename)
            if with_content:
                content_list.append(content)

    if with_content:
        return file_list, content_list, meta_data
    return file_list, meta_data

def get_upload_item_dir():
    upload_id = str(int(time.time() * 10000000))
    target_dir = os.path.join("storage", "upload", upload_id)

    # check conflict
    while os.path.exists(target_dir):
        upload_id = str(int(time.time() * 10000000))
        target_dir = os.path.join("storage", "upload", upload_id)
    return target_dir, upload_id

def get_compressed_filelist(filepath):
    file_list = []
    ext = os.path.splitext(filepath)[1]
    try:
        if ext == ".zip":
            with zipfile.ZipFile(filepath) as zf:
                unzip_file_list += zf.namelist()
        elif ext in [".gz", ".tar", ".tgz"]:
            with tarfile.TarFile(filepath) as zf:
                file_list += zf.getnames()
    except:
        print("Invalid zip file:", filepath)
    return file_list

def process_upload_item(uploadId, id, filename):
    upload_path = os.path.join("storage", "upload", uploadId)
    if os.path.exists(upload_path):
        run_path = os.path.join("storage", "run", id)
        if os.path.exists(run_path):
            shutil.rmtree(run_path)
        else:
            storage_util.ensure_path(os.path.join("storage", "run"))
        shutil.copytree(upload_path, run_path)
        zip_path = os.path.join(run_path, filename)
        ext = os.path.splitext(filename)[1]
        if ext in [".zip", ".gz", "tgz", ".tar"]:
            shutil.unpack_archive(zip_path, run_path)
    return run_path