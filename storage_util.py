import os
import json

default_storage_info = [
    {
        "name": "system",
        "id": "0",
        "path": "storage/0",
        "trait": ["system"]
    },
    {
        "name": "default",
        "id": "1",
        "path": "storage/1",
        "trait": ["user"]
    }
]

current_storage_list = None

def get_storage_info(trait):
    global current_storage_list
    if current_storage_list is None:
        storage_path = os.path.join("config", "storage.json")
        if os.path.exists(storage_path):
            with open(storage_path, "rt", encoding="utf-8") as fp:
                current_storage_list = json.load(fp)
        else:
            current_storage_list = default_storage_info

    return [x for x in current_storage_list if trait in x['trait'] or trait == "*"]

def get_storage_file_path(storage_id: str, file_path: str):
    storage_info_list = get_storage_info("*")
    storage_path = None
    for storage_info in storage_info_list:
        if storage_info['id'] == storage_id:
            storage_path = storage_info['path']
            break
    if storage_path is not None:
        file_path = os.path.normpath(file_path)
        if file_path.startswith("/") or file_path.startswith(".."):
            return storage_path  # prevent invalid access to upper directory
        return os.path.join(storage_path, file_path)
    return ""


def get_file_list(storage_id: str, file_path: str = "."):
    storage_file_path = get_storage_file_path(storage_id, file_path)
    if os.path.exists(storage_file_path):
        if os.access(storage_file_path, os.R_OK):
            file_list = os.listdir(storage_file_path)
            return file_list
    return []


def get_file_info(storage_id, file_path, file_list):
    storage_file_path = get_storage_file_path(storage_id, file_path)
    file_info_list = []
    for filename in file_list:
        filepath = os.path.join(storage_file_path, filename)
        info = {
            'name': filename,
            'is_dir': os.path.isdir(filepath),
            'size': os.path.getsize(filepath),
            'mtime': os.path.getmtime(filepath)
        }
        file_info_list.append(info)
    return file_info_list

def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)