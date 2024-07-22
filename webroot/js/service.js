import { http_get, http_post, http_put, http_delete } from "/js/http.js";
import { joinPath, createStorageFileURL } from "/js/storage_utils.js";

function makeQueryString(o) {
    let s = '?';
    for (const [key, value] of Object.entries(o)) {
        s += `${key}=${value}&`;
    }
    return s.substring(0, s.length - 1);
}

// POST object data from form elements
export function createPostItem(itemSpec) {
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

async function GetJsonHandler(url) {
    try {
        const res = await http_get(url);
        return JSON.parse(res);
    } catch(err) {
        return HTTPErrorHandler(err);
    }
}

// IMAGE
export async function createExecImage(item) {
    const res = await http_post('/api/image/create', item);
    return JSON.parse(res);
}

export async function getExecImageCreationInfo(tagName) {
    const res = await http_get(`/api/image/create/${tagName}`);
    return JSON.parse(res);
}

export async function removeExecImageCreationInfo(tagName) {
    const res = await http_delete(`/api/image/create/${tagName}`, null);
    return JSON.parse(res);
}

export async function getExecImageList() {
    const res = await http_get("/api/image/list");
    return JSON.parse(res);
}

export async function removeExecImage(tagName) {
    const res = await http_delete(`/api/image/item/${tagName}`);
    return JSON.parse(res);
}

// EXEC
export async function getExecList() {
    const res = await http_get("/api/exec/list");
    return JSON.parse(res);
}

export async function createExec(item) {
    const res = await http_post('/api/exec/create', item);
    return JSON.parse(res);
}

export async function getExecInfo(id) {
    const res = await http_get(`/api/exec/info/${id}`);
    return JSON.parse(res);
}

export async function getExecLogs(id) {
    const res = await http_get(`/api/exec/logs/${id}`);
    return JSON.parse(res);
}

export async function stopExec(id) {
    const res = await http_put(`/api/exec/stop/${id}`);
    return JSON.parse(res);
}

export async function removeExec(id) {
    const res = await http_delete(`/api/exec/item/${id}`, null);
    return JSON.parse(res);
}

export async function getExecProgress(id) {
    const res = await http_get(`/api/exec/progress/${id}`);
    return JSON.parse(res);
}

export async function getExecResult(id) {
    const res = await http_get(`/api/exec/result/${id}`);
    return JSON.parse(res);
}

// STORAGE
export async function getStorageList() {
    const res = await http_get("/api/storage/list");
    const storageList = JSON.parse(res);
    return storageList;
}

export async function getFileList(storageId, storagePath, page=0, count=0) {
    let url = joinPath("/api/storage/list", storageId, storagePath);
    const query = makeQueryString({
        page: page,
        count: count
    });
    if (count > 0) {
        url = url + query;
    }
    const res = await http_get(url);
    const fileList = JSON.parse(res);
    return fileList;
}

export async function createStorageFolder(storageId, storagePath) {
    const url = joinPath("/api/storage/dir", storageId, storagePath);
    try {
        const res = await http_put(url)
        return JSON.parse(res);
    } catch (err) {
        return HTTPErrorHandler(err);
    }
}

export async function getStorageFileContent(storageId, storagePath, reload=false) {
    const url = createStorageFileURL(storageId, storagePath, reload);
    try {
        const res = await http_get(url);
        return res;
    } catch (err) {
        return JSON.parse(err.response).detail;
    }
}

export async function deleteStorageItem(storageId, storagePath) {
    try {
        const url = joinPath("/api/storage/item", storageId, storagePath);
        const res = await http_delete(url);
        return JSON.parse(res);
    } catch(err) {
        return HTTPErrorHandler(err);
    }
}

export async function uploadFileToStorage(storageId, storagePath, contents, contentType='application/octet-stream') {
    const url = createStorageFileURL(storageId, storagePath);
    const res = await http_put(url, contents, contentType);
    return JSON.parse(res);
}

export function getUploadItemURL() {
    return "/api/storage/upload_item";
}

export async function removeUploadItem(uploadId) {
    const url = `/api/storage/upload_item/${uploadId}`;
    const res = await http_delete(url)
    return JSON.parse(res);
}

// DATASET
export async function getDatasetList() {
    const res = await http_get("/api/dataset/list");
    const datasetList = JSON.parse(res);
    return datasetList;
}

export async function saveDatasetList(datasetList) {
    const res = await http_post("/api/dataset/list", datasetList);
    return JSON.parse(res);
}

export async function addDatasetToList(dataset) {
    const res = await http_post(`/api/dataset/item/${dataset.name}`, dataset);
    return JSON.parse(res);
}

export async function updateDatasetInList(dataset) {
    const res = await http_put(`/api/dataset/item/${dataset.name}`, dataset);
    return JSON.parse(res);
}

export async function removeDatasetFromList(datasetName) {
    const res = await http_delete(`/api/dataset/item/${datasetName}`);
    return JSON.parse(res);
}

// Model
export async function getModelList() {
    const res = await http_get("/api/model/list");
    const modelList = JSON.parse(res);
    return modelList;
}

export async function saveModelList(modelList) {
    const res = await http_post("/api/model/list", modelList);
    return JSON.parse(res);
}

export async function addModelToList(model) {
    const res = await http_post(`/api/model/item/${model.name}`, model);
    return JSON.parse(res);
}

export async function updateModelInList(model) {
    const res = await http_put(`/api/model/item/${model.name}`, model);
    return JSON.parse(res);
}

export async function removeModelFromList(modelName) {
    const res = await http_delete(`/api/model/item/${modelName}`);
    return JSON.parse(res);
}

// APP
export async function getAppList() {
    const res = await http_get("/api/app/list");
    return JSON.parse(res);
}

export async function runApp(appId, params) {
    try {
        const res = await http_post(`/api/app/run/${appId}`, params);
        return JSON.parse(res);
    } catch(err) {
        return HTTPErrorHandler(err);
    }
}

export async function getAppProgress(appId) {
    return GetJsonHandler(`/api/app/progress/${appId}`);
}

export async function getAppLogs(appId) {
    return GetJsonHandler(`/api/app/logs/${appId}`)
}

export async function getAppResult(appId) {
    return GetJsonHandler(`/api/app/result/${appId}`);
}

export async function stopApp(appId) {
    return GetJsonHandler(`/api/app/stop/${appId}`);
}

export async function removeApp(appId) {
    return GetJsonHandler(`/api/app/remove/${appId}`);
}

// SYSINFO
export async function getSysInfo() {
    const res = await http_get("/api/system/info");
    return JSON.parse(res);
}

// Session
export async function createSession(username) {
    const res = await http_post(`/api/session/create/${username}`);
    return JSON.parse(res);
}

export async function getSession() {
    try {
        const res = await http_get("/api/session/current");
        return JSON.parse(res);
    } catch(err) {
        return HTTPErrorHandler(err);
    }
}

export async function deleteSession() {
    try {
        const res = await http_delete("/api/session/current");
        return JSON.parse(res);
    } catch (err) {
        return HTTPErrorHandler(err);
    }
}

export async function saveSessionData(data) {
    const res = await http_post("/api/session/data", data);
    return JSON.parse(res);
}

export async function getSessionData() {
    const res = await http_get("/api/session/data");
    return JSON.parse(res);
}

export async function deleteSessionData() {
    const res = await http_delete("/api/session/data");
    return JSON.parse(res);
}

// TENSORBOARD
export async function startTensorboard(execId) {
    const res = await http_get(`/api/tensorboard/start/${execId}`);
    return JSON.parse(res);
}

export async function stopTensorboard() {
    const res = await http_get("/api/tensorboard/stop");
    return JSON.parse(res);
}