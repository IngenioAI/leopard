from typing import Union

from pydantic import BaseModel
from fastapi import APIRouter
from fastapi_util import JSONResponseHandler
from docker_runner import DockerRunner

image_router = APIRouter(prefix="/api/image", tags=["Image"])
docker_runner = DockerRunner()


class CreateImageItem(BaseModel):
    name: str
    baseImage: str
    update: Union[bool, None] = None
    aptInstall: Union[str, None] = None
    pipInstall: Union[str, None] = None
    additionalCommand: Union[str, None] = None


@image_router.post("/create")
async def create_image(data: CreateImageItem):
    ret = docker_runner.create_image(data.name, data.baseImage, data.update, data.aptInstall, data.pipInstall,
                                     data.additionalCommand)
    return JSONResponseHandler({
        'success': ret
    })


@image_router.get("/create/{name:path}")
async def get_image_creation_info(name: str):
    info = docker_runner.get_create_image_info(name)
    return JSONResponseHandler(info)


@image_router.delete("/create/{name:path}")
async def remove_image_creation_info(name: str):
    docker_runner.remove_create_image_info(name)
    return JSONResponseHandler({
        'success': True
    })


@image_router.get("/list")
async def get_image_list():
    return JSONResponseHandler(docker_runner.list_images())


@image_router.delete("/item/{name:path}")
async def delete_image(name: str):
    res, error_info = docker_runner.remove_image(name)
    response = {"success": res}
    if error_info is not None:
        response.update(error_info)
    return JSONResponseHandler(response)
