import { getE, getV, clearE, addE, showE, hideE, addEvent, createE, createElem } from "/js/dom_utils.js";
import { createListGroupItem } from "/js/control/list_group.js";
import { ContextMenu } from "/js/control/context_menu.js";
import { runApp, getAppProgress, getAppResult, removeApp } from "/js/service.js";

import { showAskMessageBox } from "/js/dialog/ask_messagebox.js";

let modelInfoList;
let currentModelInfo;
let currentAttack;

function getModelInfo(name) {
    for (let modelInfo of modelInfoList) {
        if (modelInfo.name == name) {
            return modelInfo
        }
    }
    return null;
}

function createContextMenu(info) {
    const MENU_ID = {
        DETAIL_INFO: 0,
        DELETE: 1
    };
    const contextMenu = new ContextMenu([
            { id: MENU_ID.DETAIL_INFO, title: "상세정보", info: info },
            { id: MENU_ID.DELETE, title: "삭제", info: info},
        ],
        async (menuId, info) => {
            if (menuId == MENU_ID.DETAIL_INFO) {
                console.log(info)
            }
            else if (menuId == MENU_ID.DELETE) {
                const answer = await showAskMessageBox("정말로 삭제하시겠습니까?", "파일삭제", ["확인", "취소"]);
                if (answer.index == 0) {
                    await runApp("mia_server", { mode: "remove_model", model_name: info.name});
                    updateModelList();
                }
            }
        }
    );
    return contextMenu.element;
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
        }, createContextMenu(info));

    return item;
}

function createModelItem(modelInfo, onclick) {
    const items = modelInfo.name.split("_")
    return createListItem({
        name: modelInfo.name,
        class: "bi-layers",
        description: `${items[1]} with ${items[2]}`,
        tag: items[0],
        info: modelInfo.info
    }, onclick);
}

async function onClickModelItem(name) {
    const modelList = getE("model_list");
    clearE(modelList);
    if (name == "새로 학습") {
        addE(modelList, createListItem(
            { name: "새로 학습", description: "새로운 모델을 학습합니다.", tag: "", class: "bi-play"}));
        showE("model_train_div");
    }
    else {
        const modelInfo = getModelInfo(name);
        console.log(modelInfo);
        currentModelInfo = modelInfo;
        if (modelInfo) {
            addE(modelList, createModelItem(modelInfo));
            showE("model_attack_div");
        }
    }
}

async function updateModelList() {
    const modelListDiv = getE("model_list");
    const res = await runApp("mia_server", { mode: "list_model"});
    if (res.success) {
        clearE(modelListDiv);
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

async function drawLossGraph(progressInfo) {
    if (progressInfo.max_epochs) {
        const myChart = echarts.init(document.getElementById('train_loss_graph'));
        const option = {
            animationDuration: 500,
            animationDurationUpdate: 500,
            xAxis: {
                name: 'epoch',
                nameLocation: 'middle',
                type: 'category',
                //data: Array.from(new Array(progressInfo.max_epochs), (val, ind) => `${ind+1}`),
                data: Array.from(new Array(progressInfo.current_epoch), (val, ind) => `${ind+1}`),
            },
            yAxis: [
                {
                    name: 'loss',
                    type: 'value',
                    min: 0.0
                },
                {
                    name: 'accuracy',
                    type: 'value',
                    min: 0.0,
                    max: 100.0
                }
            ],
            series: [
                {
                    name: 'train loss',
                    data: progressInfo.train_loss,
                    smooth: true,
                    type: 'line'
                },
                {
                    name: 'train accuracy',
                    yAxisIndex: 1,
                    data: progressInfo.train_acc,
                    smooth: true,
                    type: 'line'
                },
                {
                    name: 'valid accuracy',
                    yAxisIndex: 1,
                    data: progressInfo.val_acc,
                    smooth: true,
                    type: 'line'
                }
            ],
            tooltip: {
                trigger: 'axis',
            },
            legend: {
            }
        };
        myChart.setOption(option);
    }
}

async function checkTrainProgress() {
    const progressInfo = await getAppProgress("mia_train");
    console.log(progressInfo);
    if (progressInfo.status == "none") {
        progressInfo.stage = 4;
    }
    if (progressInfo.stage) {
        getE("train_progress").style = `width: ${progressInfo.stage * 25}%`;
        let msg = "";
        if (progressInfo.stage >= 1) {
            msg += "학습을 위한 데이터를 불러오고 있습니다.<br>";
        }
        if (progressInfo.stage >= 2) {
            msg += "학습 과정을 실행하고 있습니다.<br>";
            if (progressInfo.current_epoch) {
                msg += `학습 에포크 실행 ${progressInfo.current_epoch} / ${progressInfo.max_epochs} <br>`;
            }
        }
        if (progressInfo.stage >= 3) {
            msg += "학습 모델을 평가하고 있습니다.<br>";
        }
        if (progressInfo.stage >= 4) {
            msg += "학습 모델을 저장합니다.<br>";
        }
        getE("train_output_log").innerHTML = msg;
    }
    drawLossGraph(progressInfo);
    if (progressInfo.status != "running") {
        const result = await getAppResult("mia_train");
        console.log(result);
        removeApp("mia_train");
    }
    else {
        setTimeout(checkTrainProgress, 500);
    }
}

async function execTrain() {
    const defense_method = getV("select_defense");
    const res = await runApp("mia_train", {
        model_name: getV("select_model"),
        datasets: getV("select_dataset"),
        defense: defense_method,
        batch_size: 128,
        num_workers: 5,
        epochs: parseInt(getV("train_epochs")),
        shadow_num: 1,
        early_stop: 5,
        weight_decay: 1e-4,
        lr: defense_method == 'dpsgd' ? 0.1 : 0.001,
        momentum: 0.9,
        seed: 1000
    });
    console.log(res);

    showE("train_output");
    setTimeout(checkTrainProgress, 2000);
}

async function reloadModel() {
    location.reload();
}

async function checkAttackProgress() {
    const progressInfo = await getAppProgress("mia_attack");
    console.log(progressInfo);
    let msg = "";
    if (progressInfo.status == "running") {
        msg = "멤버쉽 추론 공격을 실행 중입니다.<br>";
    }
    if (progressInfo.status == "done") {
        msg = "멤버쉽 추론 공격을 완료했습니다.<br>";
    }
    if (progressInfo.status == "none") {
        msg = "멤버쉽 추론 공격이 완료되었습니다.<br>";
    }
    if (progressInfo.message) {
        msg += `${progressInfo.message.replaceAll("\n", "<br>")}<br>`;
    }
    if (progressInfo.epoch && progressInfo.max_epochs) {
        msg += `에포크 실행중 ${progressInfo.epoch}/${progressInfo.max_epochs}<br>`;
    }
    getE("attack_output_log").innerHTML = msg;
    if (progressInfo.status != "running") {
        const result = await getAppResult("mia_attack");
        console.log(result);
        removeApp("mia_attack");
        showE("attack_graph");
        if (currentAttack == 'population' || currentAttack == 'reference') {
            drawPopulationRocGraph(result);
        }
        else if (currentAttack == 'lira+shadow') {
            showLiraShadowAttackTable(result);
        }
        else if (currentAttack == 'custom') {
            showCustomAttackTable(result);
        }
    }
    else {
        setTimeout(checkAttackProgress, 1000);
    }
}

async function execAttack() {
    const infoItems = currentModelInfo.name.split("_");
    currentAttack = getV("select_attack");
    const res = await runApp("mia_attack", {
        model_path: `/model/${currentModelInfo.name}/best_model.pt`,
        train: false,
        dp_on: false,
        n_class: infoItems[0] == "cifar100" ? 100: 10,
        attack: getV("select_attack")
    });
    console.log(res);

    showE("attack_output");
    hideE("attack_graph");

    setTimeout(checkAttackProgress, 1000);
}

function drawPopulationRocGraph(info) {
    const s_data = []
    for (let i =0; i < info.tp.length; i++) {
        const fpr = info.fp[i] / (info.tn[i] + info.fp[i])
        const tpr = info.tp[i] / (info.tp[i] + info.fn[i])
        s_data.push([fpr, tpr]);
    }
    const myChart = echarts.init(document.getElementById('attack_roc_graph'));
    const option = {
        title: {
            left: 'center',
            text: `Population Metric ROC AUC: ${(info.roc_auc).toFixed(3)}`
        },
        animationDuration: 500,
        animationDurationUpdate: 500,
        xAxis: {
            name: 'fpr',
            type: 'value',
        },
        yAxis: [
            {
                name: 'tpr',
                type: 'value',
            }
        ],
        series: [
            {
                data: s_data,
                smooth: true,
                type: 'line',
                areaStyle: {}
            }
        ],
        tooltip: {
            trigger: 'axis',
        },
        legend: {
        }
    };
    myChart.setOption(option);
}

function showCustomAttackTable(info) {
    const viewDiv = getE("attack_metric_view");
    const table = createE("table", "", { class: "table"});
    const thead = createElem({
        name: "thead",
        children: [{
            name: "tr",
            children: [
                { name: "th", text: "Attack Type", attributes: { class: "col"}},
                { name: "th", text: "AUC", attributes: { class: "col"}},
            ]
        }]
    });
    addE(table, thead);
    const tbody = createE("tbody");
    for (let key of Object.keys(info)) {
        const tr = createElem({
            name: "tr",
            children: [
                { name: "td", text: key },
                { name: "td", text: info[key].AUC }
            ]
        });
        addE(tbody, tr);
    }
    addE(table, tbody);
    addE(viewDiv, table);
    showE(viewDiv);
}

function showLiraShadowAttackTable(info) {
    const viewDiv = getE("attack_metric_view");
    const table = createE("table", "", { class: "table"});
    const thead = createElem({
        name: "thead",
        children: [{
            name: "tr",
            children: [
                { name: "th", text: "Attack Type", attributes: { class: "col"}},
                { name: "th", text: "Model", attributes: { class: "col"}},
                { name: "th", text: "AUC", attributes: { class: "col"}},
                { name: "th", text: "Advantage", attributes: { class: "col"}},
            ]
        }]
    });
    addE(table, thead);
    const tbody = createE("tbody");
    for (let shadowIndex = 0; shadowIndex < info.lira_metric.length; shadowIndex++) {
        const obj = info.lira_metric[shadowIndex];
        for (let key of Object.keys(obj)) {
            const tr = createElem({
                name: "tr",
                children: [
                    { name: "td", text: `LiRA Metric with ${key}`},
                    { name: "td", text: `Shadow Model #${shadowIndex}` },
                    { name: "td", text: obj[key].auc },
                    { name: "td", text: obj[key].advantage }
                ]
            });
            addE(tbody, tr);
        }
    }
    addE(table, tbody);
    addE(viewDiv, table);

    showCustomAttackTable({
        'Shadow Metric': { 'AUC': info.shadow_metric.roc_auc }
    })

    showE(viewDiv);
}

async function init() {
    showE("model_div");
    addEvent("btn_exec_train", "click", execTrain);
    addEvent("btn_exec_attack", "click", execAttack);
    addEvent("btn_model_reload", "click", reloadModel);

    updateModelList();
}

init();