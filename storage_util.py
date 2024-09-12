import os
import data_store

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


def get_storage_info(trait):
    storage_list = data_store.manager.get_data_list("storage")
    if len(storage_list) == 0:
        storage_list = default_storage_info
        for storage in storage_list:
            ensure_path(storage["path"])
        data_store.manager.save_data_list("storage", storage_list)
    return [x for x in storage_list if trait in x['trait'] or trait == "*"]

def get_storage_file_path(storage_id: str, file_path: str):
    storage_info_list = get_storage_info("*")
    storage_path = None
    for storage_info in storage_info_list:
        if storage_info['id'] == storage_id:
            storage_path = storage_info['path']
            break
    if storage_path is not None:
        file_path = os.path.normpath(file_path)
        if file_path.startswith("/"):
            file_path = file_path[1:]
        if file_path.startswith("../"):
            file_path = file_path[3:]
        return os.path.join(storage_path, file_path)
    return ""


def get_file_list(storage_id: str, file_path: str = ".", sort = "default"):
    storage_file_path = get_storage_file_path(storage_id, file_path)
    if os.path.exists(storage_file_path):
        if os.access(storage_file_path, os.R_OK):
            _, dir_list, file_list = next(os.walk(storage_file_path))
            if sort == "default":
                dir_list = sorted(dir_list)
                file_list = sorted(file_list)
            return dir_list + file_list
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
