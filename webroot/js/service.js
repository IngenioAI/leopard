function makeQueryString(o) {
    let s = '?';
    for (const [key, value] of Object.entries(o)) {
        s += `${key}=${value}&`;
    }
    return s.substring(0, s.length - 1);
}

// POST object data from form elements
function createPostItem(itemSpec) {
    const o = new Object();
    for (const [key, id] of itemSpec) {
        o[key] = document.getElementById(id).value;
    }
    return o;
}

function HTTPErrorHandler(err) {
    const response = JSON.parse(err.response);
    return {
        success: false,
        errorCode: err.status,
        errorMessage: response.detail
    }
}

// IMAGE
async function createExecImage(item) {
    const res = await http_post('/api/image_create', item);
    return JSON.parse(res);
}

async function getExecImageCreationInfo(tagName) {
    const res = await http_get(`/api/image_create/${tagName}`);
    return JSON.parse(res);
}

async function removeExecImageCreationInfo(tagName) {
    const res = await http_delete(`/api/image_create/${tagName}`, null);
    return JSON.parse(res);
}

async function getExecImageList() {
    const res = await http_get("/api/image/list");
    return JSON.parse(res);
}

async function removeExecImage(tagName) {
    const res = await http_delete(`/api/image/${tagName}`);
    return JSON.parse(res);
}

// EXEC
async function getExecList() {
    const res = await http_get("/api/exec_list");
    return JSON.parse(res);
}

async function createExec(item) {
    const res = await http_post('/api/exec', item);
    return JSON.parse(res);
}

async function getExecInfo(id) {
    const res = await http_get(`/api/exec/${id}`);
    return JSON.parse(res);
}

async function getExecLogs(id) {
    const res = await http_get(`/api/exec_logs/${id}`);
    return JSON.parse(res);
}

async function stopExec(id) {
    const res = await http_put(`/api/exec_stop/${id}`);
    return JSON.parse(res);
}

async function removeExec(id) {
    const res = await http_delete(`/api/exec/${id}`, null);
    return JSON.parse(res);
}

async function getExecProgress(id) {
    const res = await http_get(`/api/exec_progress/${id}`);
    return JSON.parse(res);
}

async function getExecResult(id) {
    const res = await http_get(`/api/exec_result/${id}`);
    return JSON.parse(res);
}

// STORAGE
async function getStorageList() {
    const res = await http_get("/api/storage");
    const storageList = JSON.parse(res);
    return storageList;
}

async function getFileList(storageId, storagePath, page=0, count=0) {
    let url = joinPath("/api/storage", storageId, storagePath);
    const query = makeQueryString({
        page: page,
        count: count
    });
    if (count > 0) {
        url = url + query;
    }
    const res = await http_get(url);
    const fileList = JSON.parse(res);
    sortFileList(fileList);
    return fileList;
}

async function createStorageFolder(storageId, storagePath) {
    const url = joinPath("/api/storage", storageId, storagePath);
    try {
        const res = await http_put(url)
        return JSON.parse(res);
    } catch (err) {
        return HTTPErrorHandler(err);
    }
}

async function getStorageFileContent(storageId, storagePath, reload=false) {
    const url = createStorageFileURL(storageId, storagePath, reload);
    try {
        const res = await http_get(url);
        return res;
    } catch (err) {
        return JSON.parse(err.response).detail;
    }
}

async function deleteStorageItem(storageId, storagePath) {
    try {
        const url = joinPath("/api/storage", storageId, storagePath);
        const res = await http_delete(url);
        return JSON.parse(res);
    } catch(err) {
        return HTTPErrorHandler(err);
    }
}

async function uploadFile(storageId, storagePath, contents, contentType='application/octet-stream') {
    const url = createStorageFileURL(storageId, storagePath);
    const res = await http_put(url, contents, contentType);
    return JSON.parse(res);
}

async function removeUploadItem(uploadId) {
    const url = `/api/upload_item/${uploadId}`;
    const res = await http_delete(url)
    return JSON.parse(res);
}

// DATASET
async function getDatasetList() {
    const res = await http_get("/api/dataset");
    const datasetList = JSON.parse(res);
    return datasetList;
}

async function saveDatasetList(datasetList) {
    const res = await http_post("/api/dataset", datasetList);
    return JSON.parse(res);
}

async function addDatasetToList(dataset) {
    const res = await http_post(`/api/dataset/${dataset.name}`, dataset);
    return JSON.parse(res);
}

async function removeDatasetFromList(datasetName) {
    const res = await http_delete(`/api/dataset/${datasetName}`);
    return JSON.parse(res);
}

// Model
async function getModelList() {
    const res = await http_get("/api/model");
    const modelList = JSON.parse(res);
    return modelList;
}

async function saveModelList(modelList) {
    const res = await http_post("/api/model", modelList);
    return JSON.parse(res);
}

async function addModelToList(model) {
    const res = await http_post(`/api/model/${model.name}`, model);
    return JSON.parse(res);
}

async function removeModelFromList(modelName) {
    const res = await http_delete(`/api/model/${modelName}`);
    return JSON.parse(res);
}

// APP
async function getAppList() {
    const res = await http_get("/api/app_list");
    return JSON.parse(res);
}

async function runApp(appName, params) {
    try {
        const res = await http_post(`/api/app/${appName}`, params);
        return JSON.parse(res);
    } catch(err) {
        return HTTPErrorHandler(err);
    }
}