import { getE, clearE, addE, createElem, getV, setV, addEvent } from "/js/dom_utils.js";
import { Canvas, loadImage } from "/js/canvas_utils.js";
import { ContextMenu } from "/js/control/context_menu.js";
import { createListGroupItem } from "/js/control/list_group.js";
import { runApp, getAppProgress, getAppResult, removeApp } from "/js/service.js";
import { joinPath, createStorageFileURL } from "/js/storage_utils.js";

import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";
import { showMessageBox } from "/js/dialog/messagebox.js";
import { showInputDialogBox } from "/js/dialog/input.js";
import { showAskMessageBox } from "/js/dialog/ask_messagebox.js";


let modelInfoList;
let currentModelInfo;
let currentDatasetName;

let canvas1, canvas2, info_canvas;
let scale = 1.0;
let image = null;
let faceInfo = null;

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
                    await runApp("facenet", { mode: "remove_model", model_name: info.name});
                    updateModelList();
                    ADP.hide(getE("evaluation_div"), "fade");
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
    return createListItem({
        name: modelInfo.name,
        class: modelInfo.info.params.mode == "unlearn" ? "bi-layers-half": "bi-layers",
        description: modelInfo.info.params.mode == "unlearn" ? `${modelInfo.info.params.model_name}의 언러닝 모델 (forget=${modelInfo.info.params.forget_class_index})` : "Facenet 모델",
        tag: modelInfo.info.params.dataset,
        info: modelInfo.info
    }, onclick);
}

async function onClickModelItem(name) {
    const modelList = getE("model_list");
    if (name == "새로 학습") {
        clearE(modelList);
        addE(modelList, createListItem(
            { name: "새로 학습", description: "새로운 모델을 학습합니다.", tag: "", class: "bi-play"}));
        const datasetDiv = getE("dataset_div");
        const datasetList = getE("dataset_list");
        const res = await runApp("facenet", { mode: "list_dataset"});
        if (res.success) {
            for (const datasetInfo of res.list_dataset) {
                addE(datasetList, createListItem({ name: datasetInfo, description: "dataset", tag: "VGGFace2", class: "bi-database" }, onClickDatasetItem));
            }
            //addE(datasetList, createListItem(
            //    { name: "새로 생성", description: "새로운 데이터셋을 생성합니다.", tag: "", class: "bi-play"}, onClickDatasetItem));
        }

        ADP.show(datasetDiv, "slide-right");
    }
    else {
        const modelInfo = getModelInfo(name);
        console.log(modelInfo);
        currentModelInfo = modelInfo;
        if (modelInfo) {
            clearE(modelList);
            addE(modelList, createModelItem(modelInfo));
            ADP.show(getE("evaluation_div"), "fade");
            if (modelInfo.info.params.mode == "train") {
                getE("metrics_display_train").style = "display:block";
                getE("test_acc").innerText = `${(modelInfo.info.metrics.test_acc *100).toFixed(2)}%`;
                getE("nav-menu-normal").style = "display:block";

                setV("unlearning_model_name", modelInfo.name + "_unlearn");

                setTimeout(() => {ADP.show(getE("card_test_acc"), "flip-right")}, 500);
            }
            else {
                getE("metrics_display_unlearn").style = "display:block";
                getE("nav-menu-unlearn").style = "display:block";
                const originalModel = getModelInfo(modelInfo.info.params.model_name);
                getE("before_all_acc").innerText = `${(originalModel.info.metrics.test_acc *100).toFixed(2)}%`;
                getE("before_retain_acc").innerText = `${(modelInfo.info.metrics.before_test_retain_acc *100).toFixed(2)}%`;
                getE("after_acc").innerText = `${(modelInfo.info.metrics.test_retain_acc *100).toFixed(2)}%`;
                getE("before_forget_acc").innerText = `${(modelInfo.info.metrics.before_test_forget_acc *100).toFixed(2)}%`;
                getE("after_forget_acc").innerText = `${(modelInfo.info.metrics.test_forget_acc *100).toFixed(2)}%`;

                setTimeout(() => ADP.show(getE("card_before_all_acc"), "flip-right"), 500);
                setTimeout(() => ADP.show(getE("card_before_retain_acc"), "flip-right"), 600);
                setTimeout(() => ADP.show(getE("card_after_acc"), "flip-right"), 1500);
                setTimeout(() => ADP.show(getE("card_before_forget_acc"), "flip-right"), 700);
                setTimeout(() => ADP.show(getE("card_after_forget_acc"), "flip-right"), 1600);

                setTimeout(() => drawCompareGraph(modelInfo.info.metrics), 2000);
            }
            const res = await runApp("facenet", { mode: 'load', model_name: modelInfo.name });
            if (!res.success) {
                console.error(res);
            }
        }
        else {
            console.error("model info not found", name);
        }
    }
}

async function updateModelList() {
    const modelListDiv = getE("model_list");
    const res = await runApp("facenet", { mode: "list_model"});
    if (res.success) {
        clearE(modelListDiv);
        modelInfoList = res.list_model;
        modelInfoList.sort((a, b) => {
            if (a.name < b.name) {
                return -1
            }
            else if (a.name > b.name) {
                return 1;
            }
            return 0;
        });
        for (const modelInfo of modelInfoList) {
            addE(modelListDiv, createModelItem(modelInfo, onClickModelItem));
        }
        addE(modelListDiv, createListItem(
            { name: "새로 학습", description: "새로운 모델을 학습합니다.", tag: "", class: "bi-play"}, onClickModelItem));
        ADP.show(getE("model_div"), "slide-up");
    }
    else {
        console.log(res);
    }
}

async function onClickDatasetItem(name) {
    const datasetList = getE("dataset_list");
    clearE(datasetList);
    if (name == "새로 생성") {
        addE(datasetList, createListItem(
            { name: "새로 생성", description: "새로운 데이터셋을 생성합니다.", tag: "", class: "bi-play"}));
        getE("dataset_gen_div").style = "display:block";
    }
    else {
        addE(datasetList, createListItem({ name: name, description: "dataset", tag: "VGGFace2", class: "bi-database" }));
        currentDatasetName = name;
        getE("model_train_div").style = "diplay:block";
    }
}

async function checkTrainProgress() {
    const progressInfo = await getAppProgress("facenet_train");
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
    drawLossGraph(progressInfo, 'train_loss_graph');
    if (progressInfo.status != "running") {
        const result = await getAppResult("facenet_train");
        console.log(result);
        removeApp("facenet_train");
    }
    else {
        setTimeout(checkTrainProgress, 1000);
    }
}

async function execTrain() {
    const res = await runApp("facenet_train", {
        mode: "train",
        model_name: getV("train_model_name"),
        dataset: currentDatasetName,
        epochs: parseInt(getV("train_epochs"))
    });
    console.log(res);

    getE("train_output").style = "display:block";
    setTimeout(checkTrainProgress, 2000);
}

async function drawLossGraph(progressInfo, element) {
    if (progressInfo.max_epochs) {
        const myChart = echarts.init(getE(element));
        const option = {
            animationDuration: 1000,
            animationDurationUpdate: 1000,
            xAxis: {
                name: 'epoch',
                nameLocation: 'middle',
                type: 'category',
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
                    max: 1.0
                }
            ],
            series: [
                {
                    name: 'loss',
                    data: progressInfo.loss,
                    smooth: true,
                    type: 'line'
                },
                {
                    name: 'accuracy',
                    yAxisIndex: 1,
                    data: progressInfo.acc,
                    smooth: true,
                    type: 'line'
                }
            ],
            tooltip: {
                trigger: 'axis',
            },
            legend: {
                data: ['loss', 'accuracy']
            }
        };
        myChart.setOption(option);
    }
}

async function checkDatasetGenProgress() {
    const progressInfo = await getAppProgress("vggface2_dataset");
    console.log(progressInfo);
    getE("dataset_gen_progress").style = `width: ${((progressInfo.current_count) / progressInfo.count) * 100}%`;
    if (progressInfo.current_sample) {
        getE("dataset_gen_output_log").innerHTML = `현재 처리중: ${progressInfo.current_sample}`;
    }
    else {
        getE("dataset_gen_output_log").innerHTML = `처리 준비중`;
    }
    if (progressInfo.status != "running") {
        const result = await getAppResult("vggface2_dataset");
        console.log(result);
        removeApp("vggface2_dataset");
        getE("dataset_gen_output_log").innerHTML = `완료됨`;
    }
    else {
        setTimeout(checkDatasetGenProgress, 1000);
    }
}

async function execDatasetGen() {
    const res = await runApp("vggface2_dataset", {
        name: getV("dataset_name"),
        count: parseInt(getV("class_count")),
        korean_ratio: getE(korean_only).checked ? 1.0 : 0.0
    });

    console.log(res);
    getE("dataset_gen_output").style = "display:block";
    setTimeout(checkDatasetGenProgress, 1000);
}

async function drawCompareGraph(metrics) {
    const myChart = echarts.init(document.getElementById('compare_graph'));

    const dataSeries = [];
    dataSeries.push({
        name: '모든 데이터',
        type: "bar",
        color: "#007900",
        data: [metrics.before_test_retain_acc, metrics.test_retain_acc],
    });
    dataSeries.push({
        name: 'Forget 클래스',
        type: "bar",
        color: "#a90000",
        data: [metrics.before_test_forget_acc, metrics.test_forget_acc],
    });

    const option = {
        tooltip: {
            trigger: 'axis'
        },
        legend: {
            data: ["모든 데이터", "Forget 클래스"]
        },
        toolbox: {
            show: false,
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
                data: ['언러닝 전', '언러닝 후']
            }
        ],
        yAxis: [
            {
                type: 'value'
            }
        ],
        series: dataSeries
    };

    //myChart.setOption(option, notMerge=true);
    myChart.setOption(option);
}

async function execUnlearning() {
    clearE("unlearn_output_log");
    const res = await runApp("facenet_train", {
        mode: "unlearn",
        model_name: currentModelInfo.name,
        dataset: currentModelInfo.info.params.dataset,
        unlearn_model_name: getV("unlearning_model_name"),
        forget_class_index: parseInt(getV("unlearning_class_index")),
        unlearn_epochs: parseInt(getV("unlearning_epochs"))
    });
    console.log(res);

    getE("unlearning_output").style = "display:block";
    setTimeout(checkUnlearnProgress, 2000);
}

async function checkUnlearnProgress() {
    const progressInfo = await getAppProgress("facenet_train");
    console.log(progressInfo);
    if (progressInfo.status == "none") {
        progressInfo.stage = 4;
    }
    if (progressInfo.stage) {
        getE("unlearn_progress").style = `width: ${progressInfo.stage * 25}%`;
        let msg = "";
        if (progressInfo.stage >= 1) {
            msg += "언러닝을 위한 데이터를 불러오고 있습니다.<br>";
        }
        if (progressInfo.stage >= 2) {
            msg += "언러닝 과정을 실행하고 있습니다.<br>";
            if (progressInfo.current_epoch) {
                msg += `언러닝 에포크 실행 ${progressInfo.current_epoch} / ${progressInfo.max_epochs}<br>`;
                msg += `  Loss: ${progressInfo.loss[progressInfo.current_epoch-1]}, Accuracy: ${progressInfo.acc[progressInfo.current_epoch-1]}<br>`
            }
        }
        if (progressInfo.stage >= 3) {
            msg += "언러닝 모델을 평가하고 있습니다.<br>";
        }
        if (progressInfo.stage >= 4) {
            msg += "언러닝 모델을 저장합니다.<br>";
        }
        getE("unlearn_output_log").innerHTML = msg;
    }

    drawLossGraph(progressInfo, 'unlearn_loss_graph');

    if (progressInfo.status != "running") {
        const result = await getAppResult("facenet_train");
        console.log(result);
        removeApp("facenet_train");
        getE("btn_load_unlearning").style = "display:inline";
    }
    else {
        setTimeout(checkUnlearnProgress, 1000);
    }
}

function clearFaceUI() {
    canvas1.clear();
    canvas2.clear();
    info_canvas.clear();
    faceInfo = null;
    clearE("image_view");
}

async function uploadTestData() {
    clearFaceUI();
    const res = await runApp("facenet", {
        mode: "list_random_test_data",
        dataset: currentModelInfo.info.params.dataset,
        model: currentModelInfo.name,
        class_id: "all",
        count: 20
    });
    console.log(res);
    getE("image_view").style = "display:block";
    let thumbnail_size = 160;
    for (let i= 0; i < res.random_test_data_list.length; i++) {
        let data_path = res.random_test_data_list[i];
        const data_url = `/api/app/data/facenet/${data_path}`;
        const img = createElem({name: "div", attributes: { class: "m-2", style: `display:inline-block; width:${thumbnail_size}px; height:${thumbnail_size}px;`}, children: [
            { name: "img", attributes: {class: "img-thumbnail rounded", src: data_url,
                style: "width:100%; height:100%; cursor:pointer; object-fit: contain;"},  // object-fit: cover, contain
                events: {
                    click: async () => {
                        getE("image_view").style = "display:none";
                        image = await loadImage(data_url);
                        clearFaceUI();
                        scale = canvas1.drawImageFit(image);
                        recognizeFace('test_data', data_path, true, res.labels[i]);
                    }
                }
            },
            {
                name: "div", text: res.labels[i], attributes: { class: "px-2" }
            }
        ]});
        addE("image_view", img)
    }
}

async function uploadUrl() {
    const url = await showInputDialogBox("이미지의 URL을 입력합니다.", "이미지 불러오기", async (name) => {
        if (!name) {
            showMessageBox("URL을 입력하세요", "이미지 불러오기");
            return false;
        }
        return true;
    });
    if (url) {
        const imageUrl = url;
        try {
            image = await loadImage(imageUrl);
        }
        catch (e) {
            console.log(e);
            image = null;
        }
        clearFaceUI();

        if (image) {
            scale = canvas1.drawImageFit(image);
            recognizeFace('url', url);
        }
        else {
            showMessageBox("해당 URL로 부터 이미지를 불러오는데 실패했습니다.", "에러");
        }
    }
}

async function uploadFile() {
    const res = await showFileUploadDialogBox('0', 'app/facenet/input');
    if (res.success) {
        const imagePath = res.files[0];
        const imageUrl = createStorageFileURL("0", joinPath("app/facenet/input", imagePath));
        console.log(imageUrl)
        image = await loadImage(imageUrl);
        clearFaceUI();

        scale = canvas1.drawImageFit(image);
        recognizeFace('file', imagePath);
    }
}

async function recognizeFace(type, location, noFaceBox=false, label=null) {
    if (type == 'file') {
        faceInfo = await runApp("facenet", { mode: 'inference', image_path: location });
    }
    else if (type == 'test_data') {
        faceInfo = await runApp("facenet", { mode: 'inference', test_data_path: location });
    }
    else {
        faceInfo = await runApp("facenet", { mode: 'inference', image_url: location })
    }

    console.log(faceInfo);

    if (faceInfo.boxes) {
        if (faceInfo.boxes.length == 0) {
            showMessageBox("검출된 정보가 없습니다.", "정보");
            return;
        }

        if (!noFaceBox) {
            for (const box of faceInfo.boxes) {
                drawFaceBox(box, 3, 'lime');
            }
        }
        showFaceInfo(0, label);
        return faceInfo;
    }
    else {
        showMessageBox("영상 처리 중에 오류가 발생했습니다.", "에러");
    }
    return null;
}

function recalcFaceBox(box) {
    let x = box[0] >= 0 ? box[0] : 0;
    let y = box[1] >= 0 ? box[1] : 1;
    let w = box[2] - x;
    let h = box[3] - y;
    const x_margin = w * 0.0;
    const y_margin = h * 0.0;
    x = x - x_margin;
    y = y - y_margin;
    x = Math.max(0, x);
    y = Math.max(0, y);
    w = w  + x_margin * 2;
    h = h  + y_margin * 2;
    return [x, y, w, h];
}

function drawFaceBox(box, line_width, line_color) {
    let rcBox = recalcFaceBox(box);
    canvas2.drawRect(rcBox[0] * scale, rcBox[1] * scale, rcBox[2] * scale, rcBox[3] * scale, line_width, line_color);
}

function getLabelFromId(classId, labelInfo) {
    const keys= Object.keys(labelInfo);
    for (let key of keys) {
        if (labelInfo[key] == classId) {
            return key;
        }
    }
    return "";
}

function showFaceInfo(faceIndex, label=null) {
    let infoX = image.width * scale;
    const rcBox = recalcFaceBox(faceInfo.boxes[faceIndex]);
    info_canvas.drawImage(image, rcBox[0], rcBox[1], rcBox[2], rcBox[3], infoX+50, 10, 100, rcBox[3] * 100 / rcBox[2])

    infoX = infoX + 50 + 100 + 20;

    const classId = faceInfo.max_index[faceIndex];
    const labelText = getLabelFromId(classId, faceInfo.label);
    const prediction = faceInfo.predictions[faceIndex];
    const confidence = faceInfo.face_confidence[faceIndex];
    info_canvas.fillText(infoX, 30, `예측: [${classId}] ${labelText}`);
    info_canvas.fillText(infoX, 60, `클래스 신뢰도: ${(prediction[classId]*100).toFixed(2)}%`)
    info_canvas.fillText(infoX, 90, `얼굴 검출 신뢰도: ${(confidence*100).toFixed(2)}%`)
    if (label) {
        info_canvas.fillText(infoX, 120, `정답: ${label}`)
    }
}

function onMouseUp(e) {
    const pt = mouseScreenToCanvas(e);
    if (faceInfo) {
        for (let i=0; i < faceInfo.boxes.length; i++) {
            const rcBox = recalcFaceBox(faceInfo.boxes[i]);
            for (let i = 0; i < 4; i++) {
                rcBox[i] = rcBox[i] * scale;
            }
            if (pt[0] >= rcBox[0] && pt[0] <= rcBox[0] + rcBox[2] && pt[1] >=rcBox[1] && pt[1] <= rcBox[3] + rcBox[1]) {
                info_canvas.clear();
                showFaceInfo(i);
                break;
            }
        }
    }
}

function mouseScreenToCanvas(e) {
    const ui_canvas = document.getElementById('info_canvas1');
    var offsetX = 0, offsetY = 0;
    var element = ui_canvas;
    if (element.offsetParent) {
        do {
            offsetX += element.offsetLeft;
            offsetY += element.offsetTop;
        } while((element = element.offsetParent));
    }
    let x = e.pageX - offsetX;
    let y = e.pageY - offsetY;
    return [x, y];
}

async function reloadModel() {
    window.open(`/ui/demo/unlearning.html`, "_self");
}

async function loadUnlearning() {
    const unlearningModelName = getV("unlearning_model_name")
    window.open(`/ui/demo/unlearning.html?model=${unlearningModelName}`, "_self");
}

async function init() {
    const queryParam = getQueryParam();

    addEvent("btn_exec_test_data", "click", uploadTestData);
    addEvent("btn_exec_url", "click", uploadUrl);
    addEvent("btn_exec_upload", "click", uploadFile);
    addEvent("btn_exec_unlearning", "click", execUnlearning);
    addEvent("btn_load_unlearning", "click", loadUnlearning);
    addEvent("btn_exec_train", "click", execTrain);
    addEvent("btn_exec_dataset_gen", "click", execDatasetGen);
    addEvent("btn_model_reload", "click", reloadModel);

    canvas1 = new Canvas('canvas1');
    canvas1.init(960, 540);
    canvas2 = new Canvas('canvas2');
    canvas2.init(960, 540);

    info_canvas = new Canvas('info_canvas1');
    info_canvas.init(960+540, 540);

    getE("info_canvas1").addEventListener('mouseup', onMouseUp, false);

    await updateModelList();

    if (queryParam.model) {
        onClickModelItem(queryParam.model);
    }
}

init();