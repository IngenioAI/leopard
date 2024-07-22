import { getE, clearE, getV, setV, setT, addE, createE } from "/js/dom_utils.js";
import { joinPath, splitStoragePath, createStorageFileURL } from "/js/storage_utils.js";
import { getStorageFileContent, runApp, getAppList, getAppProgress, getAppLogs, getAppResult, removeApp } from "/js/service.js";

import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";
import { createTab, showTab } from "/js/control/tab.js";

const Terminal = window.Terminal;
const getQueryParam = window.getQueryParam;

let appInfo;

async function checkProgress() {
    const progressInfo = await getAppProgress(appInfo.id);
    setT("output_progress", JSON.stringify(progressInfo, null, 4))
    if (progressInfo.status != "running") {
        const logs = await getAppLogs(appInfo.id);
        setLogs(logs.logs);
        const result = await getAppResult(appInfo.id);
        setOutput(result);
        removeApp(appInfo.id);
    }
    else {
        setTimeout(checkProgress, 1000);
    }
}

function setLogs(logs) {
    setT("output_log", logs);
    /*
    const term = new Terminal({ convertEol: true });
    term.open(getE("output_log"));
    term.resize(120, 25);
    term.write(logs);
    */
}

async function setOutput(res) {
    setV("output_text", JSON.stringify(res, null, 4));
    if (res.image_path) {
        const image = new Image();
        image.src = createStorageFileURL(outputPath[0], `${outputPath[1]}/${res.image_path}`, true);
        addE("output_data", image);
    }
    else if (res.text_path) {
        const contents = await getStorageFileContent(outputPath[0], `${outputPath[1]}/${res.text_path}`, true);
        addE("output_data", createE("pre", contents));
    }
}

async function run() {
    clearE("output_data");
    clearE("output_log");
    clearE("output_progress");
    setV("output_text", "");

    const data = JSON.parse(getV("input_text"));

    const res = await runApp(appInfo.id, data);
    if (res.container_id) {
        // run with no_wait
        setTimeout(checkProgress, 1000);
        showTab("progress");
        return;
    }
    const logs = await getAppLogs(appInfo.id);
    setLogs(logs.logs);
    setOutput(res);

    showTab("output");
}

async function uploadFile() {
    clearE("input_data");

    const inputPath = splitStoragePath(appInfo.execution.input);

    const res = await showFileUploadDialogBox(inputPath[0], inputPath[1]);
    if (res.success) {
        const image_path = joinPath(inputPath[1], res.files[0]);
        const image_url = createStorageFileURL(inputPath[0], image_path);
        const image = new Image();
        image.src = image_url;
        addE("input_data", createE("div", `"image_path": "${res.files[0]}"`));
        addE("input_data", image);
    }
}

async function init() {
    const queryParam = getQueryParam();
    const appId = queryParam.app_id;
    const appList = await getAppList();
    for (const app of appList) {
        if (app.id == appId) {
            appInfo = app;
            setT("app_title", appInfo.name);
            break;
        }
    }
    const tab = createTab([
        { id: "input", text: "입력 데이터" },
        { id: "progress", text: "Progress" },
        { id: "log", text: "출력 로그" },
        { id: "output", text: "출력 데이터" }
    ]);
    addE("tab_div", tab);

    getE("upload_file_button").addEventListener("click", uploadFile);
    getE("run_button").addEventListener("click", run);
}

init();