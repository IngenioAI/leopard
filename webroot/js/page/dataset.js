let storageList;
let datasetList;

function getStorageName(storageId) {
    for (const storageInfo of storageList) {
        if (storageInfo.id == storageId) {
            return storageInfo.name;
        }
    }
    return "";
}

function createDatasetElement(datasetInfo) {
    return createElem({
        name: "div", attributes: { class: "col" }, children: [{
            name: "div", attributes: { class: "card border-dark h-100" }, children: [
                { name: "div", attributes: { class: "card-header" }, text: datasetInfo.type },
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
                            click: (e) => {
                                let url = `/ui/storage.html?storage_id=${datasetInfo.storageId}&storage_path=${datasetInfo.storagePath}`;
                                window.open(url, "_self");
                            }
                        }
                    },
                    { name: "button", attributes: { class: "btn btn-outline-primary m-1" }, text: datasetInfo.type == "Image" ? "이미지 보기" : "데이터 보기",
                        events: {
                            click: (e) => {
                                let url = `/ui/dataview.html?storage_id=${datasetInfo.storageId}&storage_path=${datasetInfo.storagePath}&type=${datasetInfo.type}`;
                                window.open(url, "_self");
                            }
                        }
                    }]
                }
            ]
        }]
    });
}

async function createDataset() {
    const validator = (data) => {
        for (const dataset of datasetList) {
            if (data.name == dataset.name) {
                showMessageBox("동일한 이름의 데이터셋이 이미 존재합니다.", "데이터 검증");
                return false;
            }
            if (!data.storage) {
                showMessageBox("스토리지를 지정해야 합니다.", "데이터 검증");
                return false;
            }
            if (!data.storagePath) {
                showMessageBox("경로를 지정해야 합니다.", "데이터 검증");
                return false;
            }
        }
        return true;
    }
    const data = await showFormDialogBox([
        { id: "name", title: "이름", type: "text", default: "" },
        { id: "type", title: "데이터 타입", type: "select", values:["Table", "Image", "Text", "Multi"], default: "Table" },
        { id: "storage", title: "스토리지", type: "select", values: storageList.map(item => item.name), default: storageList[0].name },
        { id: "storagePath", title: "경로", type: "text" },
        { id: "description", title: "설명", type: "text" }
    ], null, "다음 정보로 데이터셋을 생성합니다", "데이터셋 생성", validator);

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
}