let datasetList;
let modelList;

let selectedDataset = [];
let selectedDataProcessor = [];
let selectedModel = [];


function addItem(selectId, selectedList, listId, icon) {
    const selectElem = getE(selectId);
    const value = selectElem.value;
    if (value != "") {
        let firstItem = false;
        if (selectedList.length == 0) {
            firstItem = true;
        }
        if (selectedList.indexOf(value) < 0) {
            selectedList.push(value);
            const listDiv = getE(listId);
            if (firstItem) {
                clearE(listDiv);
            }
            addE(listDiv, createElem({name: "div", children: [{name: "i", attributes: {class: `bi ${icon} mx-2`}}, {name: "span", text: value}]}));
        }
    }
}

function deleteItem(listId, selectedList, icon, text) {
    const listDataset = getE(listId);
    clearE(listDataset);
    addE(listDataset, createElem({name: "div", children: [{name: "i", attributes: { class: `bi ${icon} mx-2`}}, {name: "span", text: text}]}));
    selectedList.splice(0);
}

async function addDataset() {
    addItem("dataset_select", selectedDataset, "dataset_list", "bi-database");
}

async function addDataProcessor() {
    addItem("data_processor_select", selectedDataProcessor, "data_processor_list", "bi-boxes");
}

async function addModel() {
    addItem("model_select", selectedModel, "model_list", "bi-layers");
}

async function deleteDataset() {
    deleteItem("dataset_list", selectedDataset, "bi-database", "데이터셋 없음");
}

async function deleteDataProcessor() {
    deleteItem("data_processor_list", selectedDataProcessor, "bi-boxes", "데이터 처리기 없음");
}

async function deleteModel() {
    deleteItem("model_list", selectedModel, "bi-layers", "학습 모델 없음");
}

function createContextMenu(info) {
    const MENU_ID = {
        START: 100,
        STOP: 101,
        PAUSE: 102,
        DELETE: 103
    };
    const contextMenu = new ContextMenu([
            { id: MENU_ID.START, title: "시작", info: info },
            { id: MENU_ID.STOP, title: "중지", info: info },
            { id: MENU_ID.PAUSE, title: "일시중지", info: info },
            { id: MENU_ID.DELETE, title: "삭제", info: info },
        ],
        async (menuId, info) => {
            if (menuId == MENU_ID.DELETE) {
            }
        });
    return contextMenu.element;
}

async function setting() {
    console.log(selectedDataset, selectedDataProcessor, selectedModel);
    if (selectedDataset.length == 0) {
        showMessageBox("하나 이상의 데이터셋을 추가해야 합니다.", "데이터셋")
        return;
    }
    if (selectedModel.length == 0) {
        showMessageBox("하나 이상의 모델을 추가해야 합니다.", "모델");
        return;
    }
    getE("exec_table").style = "display:block";
    clearE("exec_tbody");
    for (const dataset of selectedDataset) {
        for (const processor of ["사용 안함", ...selectedDataProcessor]) {
            for (const model of selectedModel) {
                const tr = createElem({
                    name: "tr", children: [
                        { name: "td", text: dataset },
                        { name: "td", text: processor },
                        { name: "td", text: model },
                        { name: "td", text: "실행 대기" },
                    ]
                });
                const lastTD = createE("td", "", { class: "align-top" });
                addE(lastTD, createContextMenu({}));
                addE(tr, lastTD);
                addE("exec_tbody", tr);
            }
        }
    }
    getE("setting_button").className = "btn btn-outline-primary p-2";
}

async function execute() {

}

async function init() {
    datasetList = await getDatasetList();
    modelList = await getModelList();

    const selectDataset = getE("dataset_select");
    clearE(selectDataset);
    for (const dataset of datasetList) {
        const option = createE("option", dataset.name);
        addE(selectDataset, option);
        selectDataset.value = "";
    }

    const selectProcessor = getE("data_processor_select");
    const selectModel = getE("model_select");
    clearE(selectProcessor);
    clearE(selectModel);
    for(const model of modelList) {
        const option = createE("option", model.name);
        if (model.type == "Model") {
            addE(selectModel, option);
        }
        else if (model.type == "Preprocessor") {
            addE(selectProcessor, option);
        }
    }
    selectProcessor.value = "";
    selectModel.value = "";
}