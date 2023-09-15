import os


def get_storage_file_path(storage_id: str, file_path: str):
    default_path = "storage/%s" % storage_id
    if storage_id in ['0', '1']:
        storage_path = default_path
        file_path = os.path.normpath(file_path)
        if file_path.startswith("/") or file_path.startswith(".."):
            return storage_path  # prevent invalid access to upper directory
        return os.path.join(storage_path, file_path)
    return ""


def get_file_list(storage_id: str, file_path: str = "."):
    storage_file_path = get_storage_file_path(storage_id, file_path)
    if os.path.exists(storage_file_path):
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
