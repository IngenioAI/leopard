let storageList;
let modelList;

function createModelElement(modelInfo) {
    return createElem({
        name: "div", attributes: { class: "col" }, children: [{
            name: "div", attributes: { class: "card border-dark h-100" }, children: [
                { name: "div", attributes: { class: "card-header" }, text:modelInfo.type },
                { name: "div", attributes: { class: "card-body" }, children: [
                    { name: "h5", text: modelInfo.name, attributes: { class: "card-title" } },
                    { name: "p", attributes: { class: "card-text" }, children: [
                        { name: "ul", children: [
                            { name: "li", text: "storage: " + getStorageName(modelInfo.storage, storageList) },
                            { name: "li", text: modelInfo.storagePath },
                            { name: "li", text: modelInfo.mainSrc }]
                        },
                        { name: "div", text: modelInfo.description }]
                    }]
                },
                { name: "div", attributes: { class: "card-footer bg-transparent border-0" }, children: [
                    { name: "button", attributes: { class: "btn btn-outline-primary m-1" }, text: "스토리지 보기",
                        events: {
                            click: (e) => {
                                let url = `/ui/storage.html?storage_id=${getStorageId(modelInfo.storage, storageList)}&storage_path=${modelInfo.storagePath}`;
                                window.open(url, "_self");
                            }
                        }
                    },
                    { name: "button", attributes: { class: "btn btn-outline-primary m-1" }, text: "메인 소스 보기",
                        events: {
                            click: (e) => {
                                const filePath = joinPath(modelInfo.storagePath, modelInfo.mainSrc);
                                showFileView(`위치: ${modelInfo.storagePath}`, `파일보기 - ${modelInfo.mainSrc}`, getStorageId(modelInfo.storage, storageList), filePath);
                            }
                        }
                    }]
                }
            ]
        }]
    });
}

async function createModel() {
    const validator = (data) => {
        for (const model of modelList) {
            if (data.name == model.name) {
                showMessageBox("동일한 이름의 모델이 이미 존재합니다.", "데이터 검증");
                return false;
            }
        }
        return true;
    }
    const data = await showFormDialogBox([
        { id: "name", title: "이름", type: "text", default: "" },
        { id: "type", title: "타입", type: "select", values: ["Model", "Preprocessor"], default: "Model"},
        { id: "storage", title: "스토리지", type: "select", values: storageList.map(item => item.name), default: storageList[0].name },
        { id: "storagePath", title: "경로", type: "text" },
        { id: "mainSrc", title: "메인소스", type: "text" },
        { id: "family", title: "Family", type: "text" },
        { id: "description", title: "설명", type: "text" }
    ], null, "다음 정보로 모델을 등록합니다", "모델 등록", validator);
    //console.log(data);
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
}