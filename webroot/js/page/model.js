import { getE, createElem, addE, clearE } from "/js/dom_utils.js";
import { getStorageName, joinPath } from "/js/storage_utils.js";
import { getModelList, addModelToList, getStorageList, updateModelInList, removeModelFromList } from "/js/service.js";
import { ContextMenu } from "/js/control/context_menu.js";
import { showMessageBox } from "/js/dialog/messagebox.js";
import { showAskMessageBox } from "/js/dialog/ask_messagebox.js";
import { showFormDialogBox } from "/js/dialog/formdialog.js";
import { showFileView } from "/js/dialog/fileview.js";

let storageList;
let modelList;

function getModelFormInfo() {
    return [
        { id: "name", title: "이름", type: "text", default: "" },
        { id: "type", title: "타입", type: "select", values: ["Model", "Preprocessor"], default: "Model"},
        { id: "storageId", title: "스토리지", type: "select", values: storageList.map(item => item.id), default: storageList[0].id },
        { id: "storagePath", title: "경로", type: "text" },
        { id: "mainSrc", title: "메인소스", type: "text" },
        { id: "family", title: "Family", type: "text" },
        { id: "description", title: "설명", type: "text" }
    ];
}

const validator = (data) => {
    for (const model of modelList) {
        if (data.name == model.name) {
            showMessageBox("동일한 이름의 모델이 이미 존재합니다.", "데이터 검증");
            return false;
        }
    }
    return true;
}

function createModelElement(modelInfo) {
    const elem = createElem({
        name: "div", attributes: { class: "col" }, children: [{
            name: "div", attributes: { class: "card border-dark h-100" }, children: [
                { name: "div", attributes: { class: "card-header d-flex justify-content-between align-top" }, text:modelInfo.type },
                { name: "div", attributes: { class: "card-body" }, children: [
                    { name: "h5", text: modelInfo.name, attributes: { class: "card-title" } },
                    { name: "p", attributes: { class: "card-text" }, children: [
                        { name: "ul", children: [
                            { name: "li", text: "storage: " + getStorageName(modelInfo.storageId, storageList) },
                            { name: "li", text: modelInfo.storagePath },
                            { name: "li", text: modelInfo.mainSrc }]
                        },
                        { name: "div", text: modelInfo.description }]
                    }]
                },
                { name: "div", attributes: { class: "card-footer bg-transparent border-0" }, children: [
                    { name: "button", attributes: { class: "btn btn-outline-primary m-1" }, text: "스토리지 보기",
                        events: {
                            click: () => {
                                let url = `/ui/storage.html?storage_id=${modelInfo.storageId}&storage_path=${modelInfo.storagePath}`;
                                window.open(url, "_self");
                            }
                        }
                    },
                    { name: "button", attributes: { class: "btn btn-outline-primary m-1" }, text: "메인 소스 보기",
                        events: {
                            click: () => {
                                const filePath = joinPath(modelInfo.storagePath, modelInfo.mainSrc);
                                showFileView(`위치: ${modelInfo.storagePath}`, `파일보기 - ${modelInfo.mainSrc}`, modelInfo.storageId, filePath);
                            }
                        }
                    }]
                }
            ]
        }]
    });

    const MENU_ID = {
        EDIT: 100,
        DELETE: 101
    };
    const contextMenu = new ContextMenu([
            { id: MENU_ID.EDIT, title: "수정", info: modelInfo },
            { id: MENU_ID.DELETE, title: "삭제", info: modelInfo },
        ],
        async (menuId, info) => {
            if (menuId == MENU_ID.EDIT) {
                const data = await showFormDialogBox(getModelFormInfo(), info,
                    "다음 정보로 모델을 갱신합니다", "모델 수정", null);
                if (data) {
                    await updateModelInList(data);
                    refreshModelList(true);
                }
            }
            else if (menuId == MENU_ID.DELETE) {
                const answer = await showAskMessageBox("정말로 삭제하시겠습니까?", "모델 삭제", ["확인", "취소"]);
                if (answer.index == 0) {
                    await removeModelFromList(info.name);
                    refreshModelList(true);
                }
            }
        }
    );
    addE(elem.firstChild.firstChild, contextMenu.element);

    return elem;
}

async function createModel() {
    const data = await showFormDialogBox(getModelFormInfo(), null, "다음 정보로 모델을 등록합니다", "모델 등록", validator);
    if (data) {
        const cardElem = createModelElement(data);
        addE("model_list", cardElem);
        addModelToList(data);
    }
}

async function refreshModelList(reload=false) {
    if (reload) {
        modelList = await getModelList();
    }
    clearE("model_list");
    for (const modelInfo of modelList) {
        const cardElem = createModelElement(modelInfo);
        addE("model_list", cardElem);
    }
}

async function init() {
    storageList = await getStorageList();
    refreshModelList(true);
    getE("create_model_button").addEventListener("click", createModel);
}

init();