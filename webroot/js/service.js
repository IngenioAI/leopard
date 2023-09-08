// DOCKER - IMAGE

// DOCKER - CONTAINER/EXEC

// STORAGE
async function getFileList(storageId, storagePath) {
    let url = "/api/storage/" + storageId;
    if (storagePath) {
        url += (storagePath[0] != '/' ? "/" : "") + storagePath;
    }
    const res = await http_get(url)
    const fileList = JSON.parse(res);
    sortFileList(fileList);
    return fileList;
}

async function uploadFile(storageId, storagePath, contents) {
    const url = `/api/storage_file/${storageId}/${storagePath}`;
    const res = await http_put(url, contents, 'application/octet-stream');
    console.log(res);
}