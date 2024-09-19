import { createE, getE, clearE, addE, createT } from "/js/dom_utils.js";

import { ContextMenu } from "/js/control/context_menu.js";
import { createListGroupItem } from "/js/control/list_group.js";
import { runApp } from "/js/service.js";

function createContextMenu(datasetInfo) {
    const MENU_ID = {
        DELETE: 0,
        EDIT: 1
    };
    const contextMenu = new ContextMenu([
        { id: MENU_ID.DELETE, title: "삭제", info: datasetInfo },
        { id: MENU_ID.EDIT, title: "편집", info: datasetInfo }
    ],
        async (menuId, info) => {
            if (menuId == MENU_ID.DELETE) {
                const answer = await showAskMessageBox("정말로 삭제하시겠습니까?", "파일삭제", ["확인", "취소"]);
                if (answer.index == 0) {
                }
            }
            else if (menuId == MENU_ID.EDIT) {
            }
        }
    );
    return contextMenu.element;
}

function createListItem(info, onclick) {
    //const menu = createContextMenu(datasetInfo);
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
        }, /*menu*/null);

    return item;
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
        addE(modelList, createListItem({ name: name, description: "Facenet", tag: "VGGFace2" }));
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

async function init() {
    getE("model_div").style = "display:block";
    const modelList = getE("model_list");
    const res = await runApp("facenet", { mode: "list_model"});
    if (res.success) {
        for (const modelInfo of res.list_model) {
            addE(modelList, createListItem({ name: modelInfo, description: "Facenet", tag: "VGGFace2" }, onClickModelItem));
        }
        addE(modelList, createListItem(
            { name: "새로 학습", description: "새로운 모델을 학습합니다.", tag: "", class: "bi-play"}, onClickModelItem));
    }
    else {
        console.log(res);
    }
}

init();