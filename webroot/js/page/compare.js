let datasetList;
let modelList;
let execList;

async function compare() {
    for (const exec of execList) {
        const resultInfo = await getExecResult(exec.id);
    }

    const selectedDataset = [];
    let n = 0;
    let elemId = `check_dataset_${n}`;
    let elem = getE(elemId);
    while (elem) {
        if (elem.checked) {
            for (const dataset of datasetList) {
                if (dataset.elemId == elemId) {
                    selectedDataset.push(dataset);
                }
            }
        }
        n += 1;
        elemId = `check_dataset_${n}`;
        elem = getE(elemId);
    }

    const selectedModel = [];
    n = 0;
    elemId = `check_model_${n}`;
    elem = getE(elemId);
    while (elem) {
        if (elem.checked) {
            for (const model of modelList) {
                if (model.elemId == elemId) {
                    selectedModel.push(model);
                }
            }
        }
        n += 1;
        elemId = `check_model_${n}`;
        elem = getE(elemId);
    }

    const myChart = echarts.init(document.getElementById('compare_graph'));

    const dataSeries = [];
    for (const dataset of selectedDataset) {
        const modelMetrics = [];
        for (const model of selectedModel) {
            for (const exec of execList) {
                if (exec.user_data.model == model.name && exec.user_data.dataset == dataset.name) {
                    const resultInfo = await getExecResult(exec.id);
                    modelMetrics.push(resultInfo.metric);
                    break;
                }
            }
        }
        const data = {
            name: dataset.name,
            type: "bar",
            data: modelMetrics,
            markLine: {
                data: [{ type: 'average', name: 'Avg' }]
            }
        }
        dataSeries.push(data);
    }

    const option = {
        title: {
            text: '데이터 셋 모델별 성능 비교',
        },
        tooltip: {
            trigger: 'axis'
        },
        legend: {
            data: selectedDataset.map((item) => item.name)
        },
        toolbox: {
            show: true,
            feature: {
                dataView: { show: true, readOnly: false },
                magicType: { show: true, type: ['line', 'bar'] },
                restore: { show: true },
                saveAsImage: { show: true }
            }
        },
        calculable: true,
        xAxis: [
            {
                type: 'category',
                data: selectedModel.map((item) => item.name)
            }
        ],
        yAxis: [
            {
                type: 'value'
            }
        ],
        series: dataSeries
    };

    myChart.setOption(option, notMerge=true);
}

async function selectFamily() {
    const selectFamily = getE("dataset_family_select");
    const datasetDiv = getE("dataset_check_list");
    const modelDiv = getE("model_check_list");
    clearE(datasetDiv);
    let n = 0;
    for (const dataset of datasetList) {
        if (dataset.family == selectFamily.value) {
            const elemId = `check_dataset_${n}`;
            dataset.elemId = elemId;
            n += 1;
            addE(datasetDiv, createElem({
                name: "div", attributes: { class: "form-check"}, children: [
                    { name: "input", attributes: { type: "checkbox", class: "form-check-input", id: elemId, checked: "checked" }},
                    { name: "label", attributes: { for: elemId}, text: dataset.name },
                ]
            }));
        }
        else {
            dataset.elemId = "";
        }
    }
    clearE(modelDiv);
    n = 0;
    for (const model of modelList) {
        if (model.family == selectFamily.value && model.type == "Model") {
            const elemId = `check_model_${n}`;
            model.elemId = elemId;
            n += 1;
            addE(modelDiv, createElem({
                name: "div", attributes: { class: "form-check"}, children: [
                    { name: "input", attributes: { type: "checkbox", class: "form-check-input", id: elemId, checked: "checked" }},
                    { name: "label", attributes: { for: elemId}, text: model.name },
                ]
            }));
        }
        else {
            model.elemId = "";
        }
    }
}

async function init() {
    datasetList = await getDatasetList();
    modelList = await getModelList();
    execList = await getExecList();

    const familySet = new Set();
    datasetList.map((item) => familySet.add(item.family));
    const selectFamily = getE("dataset_family_select");
    clearE(selectFamily);
    for (const family of familySet) {
        const option = createE("option", family);
        addE(selectFamily, option);
        selectFamily.value = "";
    }
}