import { getE, clearE, getV, setV, setT, addE, createE } from "/js/dom_utils.js";
import { joinPath, splitStoragePath, createStorageFileURL } from "/js/storage_utils.js";
import { getStorageFileContent, runApp, getAppList } from "/js/service.js";

import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";
import { createTab, showTab } from "/js/control/tab.js";

const Terminal = window.Terminal;
const getQueryParam = window.getQueryParam;

let appInfo;

async function run() {
    clearE("output_data");
    clearE("output_log");
    setV("output_text", "");

    let automaticLog = false;
    const outputPath = splitStoragePath(appInfo.execution.output);

    const data = JSON.parse(getV("input_text"));
    if (!("with_log" in data)) {
        data["with_log"] = true;
        automaticLog = true;
    }
    const res = await runApp(appInfo.id, data)
    if ("log" in res) {
        //setT("output_log", res.log);
        const term = new Terminal({ convertEol: true });
        term.open(getE("output_log"));
        term.resize(120, 25);
        term.write(res.log);
        if (automaticLog) {
            delete res['log'];
        }
    }
    setV("output_text", JSON.stringify(res));
    if (res.image_path) {
        const image = new Image();
        image.src = createStorageFileURL(outputPath[0], `${outputPath[1]}/${res.image_path}`, true);
        addE("output_data", image);
    }
    else if (res.text_path) {
        const contents = await getStorageFileContent(outputPath[0], `${outputPath[1]}/${res.text_path}`, true);
        addE("output_data", createE("pre", contents));
    }

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
        { id: "log", text: "출력 로그" },
        { id: "output", text: "출력 데이터" }
    ]);
    addE("tab_div", tab);

    getE("upload_file_button").addEventListener("click", uploadFile);
    getE("run_button").addEventListener("click", run);
}

init();