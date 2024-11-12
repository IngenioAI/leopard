import { getE, clearE, getV, setV, setT, addE, createE } from "/js/dom_utils.js";
import { joinPath, splitStoragePath, createStorageFileURL } from "/js/storage_utils.js";
import { getStorageFileContent, runApp, getAppList, getAppProgress, getAppLogs, getAppResult, stopApp, removeApp } from "/js/service.js";

import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";
import { showAskMessageBox } from "/js/dialog/ask_messagebox.js";
import { createTab, showTab } from "/js/control/tab.js";
import { createFormGroup, getFormGroupData } from "/js/form_group.js";

const Terminal = window.Terminal;
let terminal = null;
let lastTerminalPos = 0;
const getQueryParam = window.getQueryParam;

let appInfo;

function createTerminal(divElem) {
    const terminal = new Terminal({ convertEol: true });
    terminal.open(divElem);
    terminal.resize(120, 25);
    lastTerminalPos = 0;
    return terminal;
}

function setTerminalLog(terminal, logs) {
    terminal.write(logs.substring(lastTerminalPos));
    lastTerminalPos = logs.length;
}

function resetTerminal() {
    terminal = null;
    lastTerminalPos = 0
}

async function checkProgress() {
    const progressInfo = await getAppProgress(appInfo.id);
    setT("output_progress", JSON.stringify(progressInfo, null, 4))
    const logs = await getAppLogs(appInfo.id);
    setLogs(logs.logs);
    if (progressInfo.status != "running") {
        const result = await getAppResult(appInfo.id);
        setOutput(result);
        removeApp(appInfo.id);
    }
    else {
        setTimeout(checkProgress, 1000);
    }
}

function setLogs(logs) {
    if (terminal == null) {
        terminal = createTerminal(getE("output_log"));
    }

    if (logs.length > lastTerminalPos) {
        setTerminalLog(terminal, logs);
    }
}

async function setOutput(res) {
    setV("output_text", JSON.stringify(res, null, 4));
    if (res && res.image_path) {
        const image = new Image();
        image.src = createStorageFileURL(outputPath[0], `${outputPath[1]}/${res.image_path}`, true);
        addE("output_data", image);
    }
    else if (res && res.text_path) {
        const contents = await getStorageFileContent(outputPath[0], `${outputPath[1]}/${res.text_path}`, true);
        addE("output_data", createE("pre", contents));
    }
}

async function run() {
    clearE("output_data");
    clearE("output_log");
    clearE("output_progress");
    setV("output_text", "");
    resetTerminal();

    let data = null;
    if (appInfo.input_form_spec) {
        data = getFormGroupData(appInfo.input_form_spec, null, "GT_");
    }
    else {
        data = JSON.parse(getV("input_text"));
    }

    const res = await runApp(appInfo.id, data);
    if (res.success) {
        // run with no_wait
        setTimeout(checkProgress, 2000);
        showTab("progress");
        return;
    }
    if (res.container_id) {
        const answer = await showAskMessageBox("이미 실행중인 모듈이 있습니다. 실행 중인 모듈에 연결하거나 기존 실행을 중지할 수 있습니다.", "실행 연결",
            ["기존 모듈 연결", "기존 모듈 중지", "취소"]);
        if (answer.index == 0) {
            setTimeout(checkProgress, 2000);
            showTab("progress");
            return;
        }
        else if (answer.index == 1) {
            await stopApp(appInfo.id);
            await removeApp(appInfo.id)
        }
    }
    const logs = await getAppLogs(appInfo.id);
    setLogs(logs.logs);
    setOutput(res);
    removeApp(appInfo.id);

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

    if (appInfo.input_form_spec) {
        getE("input_textarea").style = "display:none";
        getE("input_form").style = "display:block";
        const forms = createFormGroup(appInfo.input_form_spec, null, "GT_");
        addE("input_form", forms);
    }
}

init();