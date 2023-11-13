import os
import json
import shutil

from pydantic import BaseModel
from typing import Union
from fastapi import APIRouter

import docker_runner
import data_store
import upload_util
import storage_util
from fastapi_util import JSONResponseHandler

class ExecManager():
    def __init__(self) -> None:
        self.docker = docker_runner.DockerRunner()
        self.exec_list = []
        self.load()

    def load(self):
        self.exec_list = data_store.manager.get_data_list("exec")

    def save(self):
        data_store.manager.save_data_list("exec", self.exec_list)

    def get_list(self):
        current_exec_list = [self.get_info(x['id']) for x in self.exec_list]
        return current_exec_list

    def get_run_path(self, id_name):
        return os.path.join("storage", "run", id_name)

    def create_exec(self, id_name, source_path, command_line, base_image, input_path, output_path, user_data):
        run_path = self.get_run_path(id_name)
        res, info = self.docker.exec_command(source_path, command_line, base_image, input_path, output_path, { "run_path": run_path })
        if res:
            container_id = info["container_id"]
            exec_info = self.docker.exec_inspect(container_id)
            data = {
                "id": id_name,
                "base_image": base_image,
                "command_line": command_line,
                "source_path": source_path,
                "input_path": input_path,
                "output_path": output_path,
                "name": exec_info["Name"],
                "date": exec_info["Created"],
                "container_id": container_id,
                "user_data": user_data
            }
            self.exec_list.append(data)
            self.save()
            return True, data
        return False, info

    def get_info(self, exec_id):
        exec_info = {}
        for info in self.exec_list:
            if info["id"] == exec_id:
                exec_info = dict.copy(info)
                break
        if "container_id" in exec_info:
            docker_exec_info = self.docker.exec_inspect(exec_info["container_id"])
            exec_info["container"] = docker_exec_info
        return exec_info

    def get_logs(self, exec_id):
        for info in self.exec_list:
            if info["id"] == exec_id:
                return self.docker.exec_logs(info["container_id"])
        return ""

    def stop(self, exec_id):
        for info in self.exec_list:
            if info["id"] == exec_id:
                return self.docker.exec_stop(info["container_id"])
        return False, { "error_message": "ID not found: %s" % exec_id}

    def remove_exec(self, exec_id):
        for info in self.exec_list:
            if info["id"] == exec_id:
                res, error_info = self.docker.exec_remove(info["container_id"])
                if res or error_info["error_code"] == 404:
                    run_path = self.get_run_path(info["id"])
                    if os.path.exists(run_path):
                        try:
                            shutil.rmtree(run_path)
                        except:
                            pass
                    self.exec_list.remove(info)
                    self.save()
                return res, error_info
        return False, { "error_message": "ID not found: %s" % exec_id}

    def get_progress(self, exec_id):
        progress_info_path = os.path.join(self.get_run_path(exec_id), "progress.json")
        if os.path.exists(progress_info_path):
            with open(progress_info_path, "rt", encoding="utf-8") as fp:
                return json.load(fp)
        return None

    def get_result(self, exec_id):
        result_info_path = os.path.join(self.get_run_path(exec_id), "result.json")
        if os.path.exists(result_info_path):
            with open(result_info_path, "rt", encoding="utf-8") as fp:
                return json.load(fp)
        return None


exec_manager = ExecManager()

exec_router = APIRouter(prefix="/api/exec", tags=["Exec"])

@exec_router.get("/list")
async def get_exec_list():
    return JSONResponseHandler(exec_manager.get_list())

class ExecutionItem(BaseModel):
    id: str
    srcPath: str
    command: str
    imageTag: str
    inputPath: Union[str, None] = None
    outputPath: Union[str, None] = None
    uploadId: Union[str, None] = None
    userdata: Union[dict, None] = None


@exec_router.post("/create")
async def create_execution(data: ExecutionItem):
    if data.uploadId is not None:
        source_path = exec_manager.get_run_path(data.id)
        upload_util.process_upload_item(data.uploadId, source_path, data.srcPath)
    else:
        sourcePath = data.srcPath.split(":")
        source_path = storage_util.get_storage_file_path(sourcePath[0], sourcePath[1])
        print("source_path:", sourcePath, source_path)

    if data.inputPath is not None and data.inputPath != "":
        storagePath = data.inputPath.split(":")
        input_path = storage_util.get_storage_file_path(storagePath[0], storagePath[1])
    else:
        input_path = None

    if data.outputPath is not None and data.outputPath != "":
        storagePath = data.outputPath.split(":")
        output_path = storage_util.get_storage_file_path(storagePath[0], storagePath[1])
    else:
        output_path = None
    res, info = exec_manager.create_exec(data.id, source_path, data.command, data.imageTag, input_path, output_path, data.userdata)
    if res:
        return JSONResponseHandler({
            "success": True,
            "exec_info": info
        })
    else:
        return JSONResponseHandler({ "success": False }.update(info))


@exec_router.get("/info/{exec_id}")
async def get_execution_info(exec_id: str):
    info = exec_manager.get_info(exec_id)
    return JSONResponseHandler(info)


@exec_router.get("/logs/{exec_id}")
async def get_execution_logs(exec_id: str):
    logs = exec_manager.get_logs(exec_id)
    return JSONResponseHandler({
        "success": True,
        "lines": logs
    })

@exec_router.put("/stop/{exec_id}")
async def stop_execution(exec_id: str):
    res, error_info = exec_manager.stop(exec_id)
    response = { "success": res }
    if error_info is not None:
        response.update(error_info)
    return JSONResponseHandler(response)

@exec_router.delete("/item/{exec_id}")
async def remove_execution_info(exec_id: str):
    res, error_info = exec_manager.remove_exec(exec_id)
    response = { "success": res }
    if error_info is not None:
        response.update(error_info)
    return JSONResponseHandler(response)

@exec_router.get("/progress/{exec_id}")
async def get_execution_progress(exec_id: str):
    info = exec_manager.get_progress(exec_id)
    return JSONResponseHandler(info)

@exec_router.get("/result/{exec_id}")
async def get_execution_result(exec_id: str):
    info = exec_manager.get_result(exec_id)
    return JSONResponseHandler(info)
