import { getE, getV, clearE, addE, showE, hideE, addEvent, createE, createElem } from "/js/dom_utils.js";
import { runApp, getAppProgress, getAppResult, removeApp, getAppLogs } from "/js/service.js";
import { Canvas, loadImage } from "/js/canvas_utils.js";
import { joinPath, createStorageFileURL } from "/js/storage_utils.js";
import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";

const canvasList =[];

async function drawResultImages(result) {
    getE("emotion0").innerText = result.original_image_emotion;
    for (let i = 1; i <= 5; i++) {
        const url = createStorageFileURL('0', '/app/deepprivacy/output/' + result.outputs[i-1].filename, true);
        const image = await loadImage(url);
        canvasList[i].drawImageFit(image, true);
        getE("emotion"+i).innerText = result.outputs[i-1].emotion;
    }
}

async function checkProgress() {
    const progressInfo = await getAppProgress("deepprivacy");

    if (progressInfo.status != "running") {
        const result = await getAppResult("deepprivacy");
        console.log(result);
        await removeApp("deepprivacy");
        hideE("process_spinner");
        getE("output_log").innerText = "비식별화 및 감정 인식 작업이 완료되었습니다.";

        drawResultImages(result);
    }
    else {
        showE("message_div");
        getE("output_log").innerText = "비식별화 작업을 수행 중입니다.";
        showE("process_spinner");
        setTimeout(checkProgress, 1000);
    }
}

async function execDemoData() {
    clearUI();
    await runApp("deepprivacy", {
        "image_path": "demo_sample?"
    });
    setTimeout(checkProgress, 1000);

    const inputUrl = createStorageFileURL('0', '/app/deepprivacy/input/demo_sample.jpg', true)
    const image = await loadImage(inputUrl)
    canvasList[0].drawImageFit(image, true);
}

async function execUpload() {
    const res = await showFileUploadDialogBox('0', 'app/deepprivacy/input');
    if (res.success) {
        const imagePath = res.files[0];

        clearUI();
        await runApp("deepprivacy", {
            "image_path": imagePath
        });
        setTimeout(checkProgress, 1000);

        const imageUrl = createStorageFileURL("0", joinPath("app/deepprivacy/input", imagePath), true);
        const image = await loadImage(imageUrl);
        canvasList[0].drawImageFit(image, true);
    }
}

async function onShowEvalPage() {
    const result = await runApp("deepprivacy-eval", {});
    console.log(result)

    getE("class2_before_acc").innerText = `${result.class_2_before_acc.toFixed(2)}%`;
    setTimeout(() => ADP.show(getE("card_class2_before_acc"), "flip-right"), 100);
    getE("class2_after_acc").innerText = `${result.class_2_after_acc.toFixed(2)}%`;
    setTimeout(() => ADP.show(getE("card_class2_after_acc"), "flip-right"), 1000);

    getE("class8_before_acc").innerText = `${result.class_8_before_acc.toFixed(2)}%`;
    setTimeout(() => ADP.show(getE("card_class8_before_acc"), "flip-right"), 100);

    getE("class8_after_acc").innerText = `${result.class_8_after_acc.toFixed(2)}%`;
    setTimeout(() => ADP.show(getE("card_class8_after_acc"), "flip-right"), 1000);
}

async function clearUI() {
    for (let i=0; i<6; i++) {
        canvasList[i].clear();
        clearE("emotion" + i);
    }
    clearE("output_log");
}

async function init() {
    addEvent("btn_exec_demo_data", "click", execDemoData);
    addEvent("btn_exec_upload", "click", execUpload);

    addEvent("nav-dpeval-tab", "shown.bs.tab", onShowEvalPage);

    let canvas = new Canvas('canvas0');
    canvas.init(360, 300);
    canvasList.push(canvas);

    for (let i = 1; i <= 5; i++) {
        let canvas = new Canvas('canvas' + i);
        canvas.init(320, 280);
        canvasList.push(canvas);
    }
}

init();