from fastapi.testclient import TestClient
import pytest
import time
from server import app, init_app, deinit_app

client = TestClient(app)

@pytest.fixture(scope="module")
def server():
    init_app(app)
    yield
    deinit_app(app)

def test_root(server):
    response = client.get("/")
    assert response.status_code == 200

def test_image(server):
    response = client.get("/api/image/list")
    assert response.status_code == 200
    res = response.json()
    print(res)
    for image_info in res:
        if "test-pytest:latest" in image_info["RepoTags"]:
            response = client.delete("/api/image/item/test-pytest")

    response = client.post(
        "/api/image/create",
        json={"name": "test-pytest", "baseImage": "python:3.8", "update": False, "aptInstall": "", "pipInstall": "pytest", "additionalCommand": ""}
    )
    res = response.json()
    assert response.status_code == 200
    assert res["success"]
    response = client.get("/api/image/create/test-pytest")
    res = response.json()
    assert response.status_code == 200
    assert res["status"] == "running"
    while res["status"] == "running" and response.status_code == 200:
        time.sleep(0.5)
        response = client.get("/api/image/create/test-pytest")
        res = response.json()

    assert res["status"] == "exited"
    assert res["lines"] is not None

    response = client.delete("/api/image/create/test-pytest")
    assert response.status_code == 200

    response = client.get("/api/image/list")
    assert response.status_code == 200
    res = response.json()
    image_exist = False
    for image_info in res:
        if "test-pytest:latest" in image_info["RepoTags"]:
            image_exist = True
            break
    assert image_exist

    response = client.delete("/api/image/item/test-pytest")
    assert response.status_code == 200

    response = client.get("/api/image/list")
    assert response.status_code == 200
    res = response.json()
    image_exist = False
    for image_info in res:
        if "test-pytest:latest" in image_info["RepoTags"]:
            image_exist = True
            break
    assert not image_exist

def test_storage(server):
    response = client.get("/api/storage/list")
    assert response.status_code == 200
    res = response.json()
    assert len(res) > 0
    storageId = res[0]["id"]
    assert storageId is not None

    response = client.get(f'/api/storage/list/{storageId}')
    res = response.json()
    assert response.status_code == 200

    dir_exist = False
    for item in res["items"]:
        if item["name"] == "test-pytest":
            dir_exist = True
            break
    if not dir_exist:
        response = client.put(f'/api/storage/dir/{storageId}/test-pytest')
        assert response.status_code == 200

    # Put file item
    response = client.put(f'/api/storage/file/{storageId}/test-pytest/test.txt',
               content="sample text")
    assert response.status_code == 200

    response = client.get(f'/api/storage/file/{storageId}/test-pytest/test.txt')
    res = response.content.decode("utf-8")
    assert res == "sample text"

    response = client.delete(f'/api/storage/item/{storageId}/test-pytest/test.txt')
    assert response.status_code == 200

    response = client.get(f'/api/storage/file/{storageId}/test-pytest/test.txt')
    assert response.status_code == 404

    response = client.delete(f'/api/storage/item/{storageId}/test-pytest')
    assert response.status_code == 200

    response = client.get(f'/api/storage/list/{storageId}')
    res = response.json()
    assert response.status_code == 200

    dir_exist = False
    for item in res["items"]:
        if item["name"] == "test-pytest":
            dir_exist = True
            break
    assert not dir_exist

def test_exec(server):
    response = client.get("/api/storage/list")
    assert response.status_code == 200
    res = response.json()
    assert len(res) > 0
    storageId = res[0]["id"]
    response = client.put(f'/api/storage/dir/{storageId}/test-pytest')
    response = client.put(f'/api/storage/file/{storageId}/test-pytest/test.py',
               content="print('Hello')")
    assert response.status_code == 200

    response = client.post('/api/exec/create',
                json={
                    "id": "pytest",
                    "srcPath": f'{storageId}:/test-pytest',
                    "command": "python test.py",
                    "imageTag": "python:3.8",
                    "inputPath": "",
                    "outputPath": ""
                })
    res = response.json()
    print(res)
    assert response.status_code == 200
    assert res["success"]
    exec_id = "pytest"

    response = client.get(f'/api/exec/info/{exec_id}')
    res = response.json()
    print(res)
    while res["container"]["State"]["Running"]:
        time.sleep(0.5)
        response = client.get(f'/api/exec/info/{exec_id}')
        res = response.json()

    response = client.delete(f'/api/storage/item/{storageId}/test-pytest/test.py')
    assert response.status_code == 200
    response = client.delete(f'/api/storage/item/{storageId}/test-pytest')
    assert response.status_code == 200

def test_sys_info(server):
    response = client.get("/api/system/info")
    res = response.json()
    print(res)
    assert res["cpu_info"] is not None
    assert response.status_code == 200