import { createE, getE, clearE, addE, createT } from "/js/dom_utils.js";

import { createListGroupItem } from "/js/control/list_group.js";
import { runApp } from "/js/service.js";
import { joinPath, createStorageFileURL } from "/js/storage_utils.js";

import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";
import { showMessageBox } from "/js/dialog/messagebox.js";
import { showInputDialogBox } from "/js/dialog/input.js";


let modelInfoList;

function getModelInfo(name) {
    for (let modelInfo of modelInfoList) {
        if (modelInfo.name == name) {
            return modelInfo
        }
    }
    return null;
}

function createListItem(info, onclick) {
    const item = createListGroupItem([
        {
            name: "h5", attributes: { class: "mb-1" }, children: [
                { name: "span", attributes: { class: info.class ? info.class : "bi-layers", style: "padding-right: 0.2em" } },
                { name: "span", text: info.name }
            ]
        },
        {
            name: "div", attributes: { class: "d-flex w-100 justify-content-between" }, children: [
                { name: "small", text: (info.description) ? info.description : " " },
                { name: "small", text: (info.tag) ? info.tag :  " " }
            ]
        }],
        () => {
            if (onclick) {
                onclick(info.name);
            }
        });

    return item;
}

function createModelItem(modelInfo, onclick) {
    return createListItem({
        name: modelInfo.name,
        class: modelInfo.info.params.mode == "unlearn" ? "bi-layers-half": "bi-layers",
        description: modelInfo.info.params.mode == "unlearn" ? "Facenet Unlearn 모델" : "Facenet 모델",
        tag: modelInfo.info.params.dataset
    }, onclick);
}

async function onClickModelItem(name) {
    const modelList = getE("model_list");
    clearE(modelList);
    if (name == "새로 학습") {
        addE(modelList, createListItem(
            { name: "새로 학습", description: "새로운 모델을 학습합니다.", tag: "", class: "bi-play"}));
        const datasetDiv = getE("dataset_div");
        datasetDiv.style = "display:block";
        const datasetList = getE("dataset_list");
        const res = await runApp("facenet", { mode: "list_dataset"});
        if (res.success) {
            for (const datasetInfo of res.list_dataset) {
                addE(datasetList, createListItem({ name: datasetInfo, description: "dataset", tag: "VGGFace2", class: "bi-database" }, onClickDatasetItem));
            }
            addE(datasetList, createListItem(
                { name: "새로 생성", description: "새로운 데이터셋을 생성합니다.", tag: "", class: "bi-play"}, onClickDatasetItem));
        }
    }
    else {
        const modelInfo = getModelInfo(name);
        if (modelInfo) {
            addE(modelList, createModelItem(modelInfo));
            getE("evaluation_div").style = "display:block";
            const metricsDiv = getE("metrics_display_div");
            if (modelInfo.info.params.mode == "train") {
                metricsDiv.innerHTML = `Test 정확도: ${(modelInfo.info.metrics.test_acc *100).toFixed(2)}%`;
            }
            else {
                metricsDiv.innerHTML = `Test 정확도 (Forget): ${(modelInfo.info.metrics.test_forget_acc *100).toFixed(2)}%` + "<br>" +
                        `Test 정확도 (Retain): ${(modelInfo.info.metrics.test_retain_acc *100).toFixed(2)}%`
                const originalModel = getModelInfo(modelInfo.info.params.model_name);
                if (originalModel.info.params.mode == "train") {
                    metricsDiv.innerHTML += "<br>" + `Unlearning 이전 모델의 Test 정확도: ${(originalModel.info.metrics.test_acc *100).toFixed(2)}%`;
                }
            }

        }
        else {
            console.error("model info not found", name);
        }
    }
}

async function onClickDatasetItem(name) {
    const datasetList = getE("dataset_list");
    clearE(datasetList);
    if (name == "새로 생성") {
        addE(datasetList, createListItem(
            { name: "새로 생성", description: "새로운 데이터셋을 생성합니다.", tag: "", class: "bi-play"}));
    }
    else {
        addE(datasetList, createListItem({ name: name, description: "dataset", tag: "VGGFace2", class: "bi-database" }));
    }
}

async function uploadTestData() {

}

async function uploadUrl() {
    const url = await showInputDialogBox("이미지의 URL을 입력합니다.", "이미지 불러오기", async (name) => {
        if (!name) {
            showMessageBox("URL을 입력하세요", "이미지 불러오기");
            return false;
        }
        return true;
    });
    if (url) {
        console.log(url);
    }
}

async function uploadFile() {
    const res = await showFileUploadDialogBox('0', 'app/facenet/input');
    if (res.success) {
        const imagePath = res.files[0];
        const imageUrl = createStorageFileURL("0", joinPath("app/facenet/input", imagePath));
        console.log(imageUrl)
        //recognizeFace('file', imagePath);
    }
}

async function init() {
    getE("model_div").style = "display:block";
    getE("btn_exec_test_data").addEventListener("click", uploadTestData);
    getE("btn_exec_url").addEventListener("click", uploadUrl);
    getE("btn_exec_upload").addEventListener("click", uploadFile);

    const modelListDiv = getE("model_list");
    const res = await runApp("facenet", { mode: "list_model"});
    if (res.success) {
        modelInfoList = res.list_model;
        for (const modelInfo of modelInfoList) {
            addE(modelListDiv, createModelItem(modelInfo, onClickModelItem));
        }
        addE(modelListDiv, createListItem(
            { name: "새로 학습", description: "새로운 모델을 학습합니다.", tag: "", class: "bi-play"}, onClickModelItem));
    }
    else {
        console.log(res);
    }
}

init();