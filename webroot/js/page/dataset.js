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
                {
                    name: "div", attributes: { class: "card-body" }, children: [
                        { name: "h5", text: datasetInfo.name, attributes: { class: "card-title" } },
                        {
                            name: "p", attributes: { class: "card-text" }, children: [
                                {
                                    name: "ul", children: [
                                        { name: "li", text: "storage: " + getStorageName(datasetInfo.storageId) },
                                        { name: "li", text: datasetInfo.storagePath }
                                    ]
                                },
                                { name: "div", text: datasetInfo.description }
                            ]
                        }
                    ]
                },
                {
                    name: "div", attributes: { class: "card-footer bg-transparent border-0" }, children: [
                        {
                            name: "button", attributes: { class: "btn btn-outline-primary m-1" }, text: "스토리지 보기",
                            events: {
                                click: (e) => {
                                    let url = `/ui/storage.html?storage_id=${datasetInfo.storageId}&storage_path=${datasetInfo.storagePath}`;
                                    window.open(url, "_self");
                                }
                            }
                        },
                        {
                            name: "button", attributes: { class: "btn btn-outline-primary m-1" }, text: datasetInfo.type == "Image" ? "이미지 보기" : "데이터 보기",
                            events: {
                                click: (e) => {
                                    let url = `/ui/dataview.html?storage_id=${datasetInfo.storageId}&storage_path=${datasetInfo.storagePath}`;
                                    window.open(url, "_self");
                                }
                            }
                        }
                    ]
                }
            ]
        }
        ]
    });
}

async function init() {
    storageList = await getStorageList();
    datasetList = await getDatasetList();
    for (const datasetInfo of datasetList) {
        const cardElem = createDatasetElement(datasetInfo);
        addE(getE("card_container"), cardElem);
    }
}