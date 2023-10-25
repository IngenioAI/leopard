let datasetList;
let modelList;

function compare() {
    const myChart = echarts.init(document.getElementById('compare_graph'));

    const option = {
        title: {
            text: 'MNIST 데이터 셋 모델별 성능 비교',
            subtext: 'Fake Data'
        },
        tooltip: {
            trigger: 'axis'
        },
        legend: {
            data: ['MNIST', 'MNIST-N10']
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
                // prettier-ignore
                data: ['MNIST-CNN', 'MNIST-MLP']
            }
        ],
        yAxis: [
            {
                type: 'value'
            }
        ],
        series: [
            {
                name: 'MNIST',
                type: 'bar',
                data: [
                    98.7, 97.7
                ],
                markPoint: {
                    data: [
                        { type: 'max', name: 'Max' },
                        { type: 'min', name: 'Min' }
                    ]
                },
                markLine: {
                    data: [{ type: 'average', name: 'Avg' }]
                }
            },
            {
                name: 'MNIST-N10',
                type: 'bar',
                data: [
                    97.1, 93.1
                ],
                markPoint: {
                    data: [
                        { type: 'max', name: 'Max' },
                        { type: 'min', name: 'Min' }
                    ]
                },
                markLine: {
                    data: [{ type: 'average', name: 'Avg' }]
                }
            }
        ]
    };

    myChart.setOption(option);
}

async function selectFamily() {
    const selectFamily = getE("dataset_family_select");
    const datasetDiv = getE("dataset_check_list");
    const modelDiv = getE("model_check_list");
    clearE(datasetDiv);
    for (const dataset of datasetList) {
        if (dataset.family == selectFamily.value) {
            addE(datasetDiv, createElem({
                name: "div", attributes: { class: "form-check"}, children: [
                    { name: "input", attributes: { type: "checkbox", class: "form-check-input", id: `check_${dataset.name}` }},
                    { name: "label", attributes: { for: `check_${dataset.name}`}, text: dataset.name },
                ]
            }));
        }
    }
    clearE(modelDiv);
    for (const model of modelList) {
        if (model.family == selectFamily.value && model.type == "Model") {
            addE(modelDiv, createElem({
                name: "div", attributes: { class: "form-check"}, children: [
                    { name: "input", attributes: { type: "checkbox", class: "form-check-input", id: `check_${model.name}` }},
                    { name: "label", attributes: { for: `check_${model.name}`}, text: model.name },
                ]
            }));
        }
    }
}

async function init() {
    datasetList = await getDatasetList();
    modelList = await getModelList();
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