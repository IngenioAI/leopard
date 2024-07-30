import { Canvas, loadImage } from "/js/canvas_utils.js";
import { getE, isJSONEmpty } from "/js/dom_utils.js";
import { joinPath, createStorageFileURL } from "/js/storage_utils.js";
import { runApp, createStorageFolder } from "/js/service.js";

import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";
import { showMessageBox } from "/js/dialog/messagebox.js";
import { showInputDialogBox } from "/js/dialog/input.js";

let canvas1, canvas2, info_canvas;
let scale = 1.0;
let image = null;
let faceInfo = null;

async function loadModel() {
    const modelName = await showInputDialogBox("불러올 모델의 이름을 입력합니다.", "모델 불러오기", async (name) => {
        if (!name) {
            showMessageBox("모델 이름을 입력하세요", "모델 불러오기");
            return false;
        }
        return true;
    });

    if (modelName) {
        const res = await runApp("facenet", { mode: 'load', model_name: modelName });
        console.log(res);
        if (res.success) {
            showMessageBox("모델 불러오기가 완료되었습니다.", "모델 불러오기");
        }
        else {
            showMessageBox("모델 불러오기에 실패했습니다.<br/> " + res.message, "모델 불러오기");
        }
    }
}

async function uploadFile() {
    const res = await showFileUploadDialogBox('0', 'app/facenet/input');
    if (res.success) {
        const imagePath = res.files[0];
        const imageUrl = createStorageFileURL("0", joinPath("app/facenet/input", imagePath));
        image = await loadImage(imageUrl);
        canvas1.clear();
        canvas2.clear();
        info_canvas.clear();
        scale = canvas1.drawImageFit(image);

        recognizeFace('file', imagePath);
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
        image = await loadImage(imageUrl);
        canvas1.clear();
        canvas2.clear();
        info_canvas.clear();
        scale = canvas1.drawImageFit(image);

        recognizeFace('url', url);
    }
}

async function recognizeFace(type, location) {
    if (type == 'file') {
        faceInfo = await runApp("facenet", { mode: 'inference', image_path: location });
    }
    else {
        faceInfo = await runApp("facenet", { mode: 'inference', image_url: location })
    }

    console.log(faceInfo);

    if (faceInfo.boxes.length == 0) {
        showMessageBox("검출된 정보가 없습니다.", "정보");
        return;
    }

    for (const box of faceInfo.boxes) {
        drawFaceBox(box, 3, 'lime');
    }

    showFaceInfo(0);

    return faceInfo;
}

function recalcFaceBox(box) {
    let x = box[0] >= 0 ? box[0] : 0;
    let y = box[1] >= 0 ? box[1] : 1;
    let w = box[2] - x;
    let h = box[3] - y;
    const x_margin = w * 0.2;
    const y_margin = h * 0.2;
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

function showFaceInfo(faceIndex) {
    const rcBox = recalcFaceBox(faceInfo.boxes[faceIndex]);
    info_canvas.drawImage(image, rcBox[0], rcBox[1], rcBox[2], rcBox[3], 10, 550, 100, rcBox[3] * 100 / rcBox[2])

    const classId = faceInfo.max_index[faceIndex];
    const labelText = getLabelFromId(classId, faceInfo.label);
    const prediction = faceInfo.predictions[faceIndex];
    const confidence = faceInfo.face_confidence[faceIndex];
    info_canvas.fillText(120, 570, `[${classId}] ${labelText}`);
    info_canvas.fillText(140, 590, `class confidence: ${(prediction[classId]*100).toFixed(2)}%`)
    info_canvas.fillText(140, 610, `detect confidence: ${(confidence*100).toFixed(2)}%`)
}

function onMouseUp(e) {
    const pt = mouseScreenToCanvas(e);
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

function init() {
    canvas1 = new Canvas('canvas1');
    canvas1.init(960, 540);
    canvas2 = new Canvas('canvas2');
    canvas2.init(960, 540);

    info_canvas = new Canvas('info_canvas1');
    info_canvas.init(960, 540+320);

    getE("fileupload_upload").addEventListener("click", uploadFile);
    getE("url_upload").addEventListener("click", uploadUrl);
    getE("model_load").addEventListener("click", loadModel);
    getE("info_canvas1").addEventListener('mouseup', onMouseUp, false);
}

init();