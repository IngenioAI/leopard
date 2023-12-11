import { Canvas, loadImage } from "/js/canvas_utils.js";
import { getE, isJSONEmpty } from "/js/dom_utils.js";
import { joinPath, createStorageFileURL } from "/js/storage_utils.js";
import { runApp } from "/js/service.js";

import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";
import { showMessageBox } from "/js/dialog/messagebox.js";

let canvas1, canvas2;
let scale = 1.0;
let image_path = undefined;

async function uploadFile() {
    const res = await showFileUploadDialogBox('0', 'app/mtcnn/data');
    if (res.success) {
        image_path = res.files[0];
        const image_url = createStorageFileURL("0", joinPath("app/mtcnn/data", image_path));
        const image = await loadImage(image_url);
        canvas1.clear();
        canvas2.clear();
        scale = canvas1.drawImageFit(image);
    }
}

async function detectFace() {
    if (!image_path) {
        showMessageBox("이미지를 먼저 업로드해 주세요", "정보");
        return;
    }
    const faceList = await runApp("mtcnn", { mode: 'detect', image_path: image_path });

    if (isJSONEmpty(faceList)) {
        showMessageBox("검출된 정보가 없습니다.", "정보");
        return;
    }

    for (const face of faceList) {
        console.log(face.box, face.confidence);
        canvas2.drawRect(face.box[0] * scale, face.box[1] * scale,
            face.box[2] * scale, face.box[3] * scale, 3, 'lime');
    }
}
async function anonymizeFace() {
    if (!image_path) {
        showMessageBox("이미지를 먼저 업로드해 주세요", "정보");
        return;
    }
    const result = await runApp("mtcnn", { mode: 'anonymize', image_path: image_path });

    if (isJSONEmpty(result)) {
        showMessageBox("검출된 정보가 없습니다.", "정보");
        return;
    }

    const img_name = result.image_path;
    image_path = img_name;
    const image_url = createStorageFileURL("0", joinPath("app/mtcnn/data/", img_name));
    const image = await loadImage(image_url);
    canvas1.clear();
    scale = canvas1.drawImageFit(image);
}


function init() {
    canvas1 = new Canvas('canvas1');
    canvas1.init(1200, 700);
    canvas2 = new Canvas('canvas2');
    canvas2.init(1200, 700);

    getE("fileupload_upload").addEventListener("click", uploadFile);
    getE("face_get").addEventListener("click", detectFace);
    getE("face_blur").addEventListener("click", anonymizeFace);
}

init();