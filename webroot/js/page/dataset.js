import { createElem, clearE, addE, getE } from "/js/dom_utils.js";
import { addDatasetToList, getDatasetList, getStorageList, updateDatasetInList, removeDatasetFromList } from "/js/service.js";
import { showMessageBox } from "/js/dialog/messagebox.js";
import { showAskMessageBox } from "/js/dialog/ask_messagebox.js";
import { showFormDialogBox } from "/js/dialog/formdialog.js";
import { ContextMenu } from "/js/control/context_menu.js";

let storageList;
let datasetList;

function getDatasetFormInfo()
{
    return [
        { id: "name", title: "이름", type: "text", default: "" },
        { id: "type", title: "데이터 타입", type: "select", values:["Table", "Image", "Text", "Multi"], default: "Table" },
        { id: "storage", title: "스토리지", type: "select", values: storageList.map(item => item.name), default: storageList[0].name },
        { id: "storagePath", title: "경로", type: "text" },
        { id: "family", title: "Family", type: "text" },
        { id: "description", title: "설명", type: "text" }
    ];
}

function validateDataset(dataset) {
    if (!dataset.storage) {
        showMessageBox("스토리지를 지정해야 합니다.", "데이터 검증");
        return false;
    }
    if (!dataset.storagePath) {
        showMessageBox("경로를 지정해야 합니다.", "데이터 검증");
        return false;
    }
    return true;
}

const datasetValidator = (data) => {
    for (const dataset of datasetList) {
        if (data.name == dataset.name) {
            showMessageBox("동일한 이름의 데이터셋이 이미 존재합니다.", "데이터 검증");
            return false;
        }
    }
    return validateDataset(data);
}

const datasetValidatorForUpdater = (data) => {
    return validateDataset(data);
}

function getStorageName(storageId) {
    for (const storageInfo of storageList) {
        if (storageInfo.id == storageId) {
            return storageInfo.name;
        }
    }
    return "";
}

function createDatasetElement(datasetInfo) {
    const elem = createElem({
        name: "div", attributes: { class: "col" }, children: [{
            name: "div", attributes: { class: "card border-dark h-100" }, children: [
                { name: "div", attributes: { class: "card-header d-flex justify-content-between align-top" }, text: datasetInfo.type },
                { name: "div", attributes: { class: "card-body" }, children: [
                    { name: "h5", text: datasetInfo.name, attributes: { class: "card-title" } },
                    { name: "p", attributes: { class: "card-text" }, children: [
                        { name: "ul", children: [
                            { name: "li", text: "storage: " + getStorageName(datasetInfo.storageId) },
                            { name: "li", text: datasetInfo.storagePath }]
                        },
                        { name: "div", text: datasetInfo.description }]
                    }]
                },
                { name: "div", attributes: { class: "card-footer bg-transparent border-0" }, children: [
                    { name: "button", attributes: { class: "btn btn-outline-primary m-1" }, text: "스토리지 보기",
                        events: {
                            click: () => {
                                let url = `/ui/storage.html?storage_id=${datasetInfo.storageId}&storage_path=${datasetInfo.storagePath}`;
                                window.open(url, "_self");
                            }
                        }
                    },
                    { name: "button", attributes: { class: "btn btn-outline-primary m-1" }, text: datasetInfo.type == "Image" ? "이미지 보기" : "데이터 보기",
                        events: {
                            click: () => {
                                if (datasetInfo.type == "Image") {
                                    let url = `/ui/imageview.html?storage_id=${datasetInfo.storageId}&storage_path=${datasetInfo.storagePath}&type=${datasetInfo.type}`;
                                    window.open(url, "_self");
                                }
                                else {
                                    let url = `/ui/dataview.html?storage_id=${datasetInfo.storageId}&storage_path=${datasetInfo.storagePath}&type=${datasetInfo.type}`;
                                    window.open(url, "_self");
                                }
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
            { id: MENU_ID.EDIT, title: "수정", info: datasetInfo },
            { id: MENU_ID.DELETE, title: "삭제", info: datasetInfo },
        ],
        async (menuId, info) => {
            if (menuId == MENU_ID.EDIT) {
                const data = await showFormDialogBox(getDatasetFormInfo(), info,
                    "다음 정보로 데이터셋을 갱신합니다", "데이터셋 수정", datasetValidatorForUpdater);
                if (data) {
                    await updateDatasetInList(data);
                    refreshDataset(true);
                }
            }
            else if (menuId == MENU_ID.DELETE) {
                const answer = await showAskMessageBox("정말로 삭제하시겠습니까?", "데이터셋 삭제", ["확인", "취소"]);
                if (answer.index == 0) {
                    await removeDatasetFromList(info.name);
                    refreshDataset(true);
                }
            }
        }
    );
    addE(elem.firstChild.firstChild, contextMenu.element);
    return elem;
}

async function createDataset() {
    const data = await showFormDialogBox(getDatasetFormInfo(), null, "다음 정보로 데이터셋을 생성합니다", "데이터셋 생성", datasetValidator);
    if (data) {
        let storageId;
        for (const storage of storageList) {
            if (storage.name == data.storage) {
                storageId = storage.id;
            }
        }
        delete data['storage'];
        data['storageId'] = storageId;
        addDatasetToList(data);
        refreshDataset(true);
    }
}

async function refreshDataset(reload=false) {
    if (reload) {
        datasetList = await getDatasetList();
    }
    clearE("card_container");
    for (const datasetInfo of datasetList) {
        const cardElem = createDatasetElement(datasetInfo);
        addE(getE("card_container"), cardElem);
    }
}

async function init() {
    storageList = await getStorageList();
    refreshDataset(true);
    getE("create_button").addEventListener("click", createDataset);
}

init();