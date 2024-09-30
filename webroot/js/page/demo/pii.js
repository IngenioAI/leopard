import { getE, clearE, addE, createElem, getV, setV, addEvent } from "/js/dom_utils.js";
import { joinPath, createStorageFileURL } from "/js/storage_utils.js";
import { getStorageFileContent } from "/js/service.js";
import { runApp, getAppResult } from "/js/service.js";

import { showMessageBox } from "/js/dialog/messagebox.js";
import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";

async function useSynthData() {

}

async function uploadFile() {
    const res = await showFileUploadDialogBox('0', 'app/pii/input');
    if (res.success) {
        const fileName = res.files[0];
        const result = await runApp('pii', {
            module: "presidio",
            action_type : 'analyze',
            text_file: fileName
        });
        console.log(result)

        let fileContent = await getStorageFileContent("0", joinPath("app/pii/input", fileName));
        let curPos = 0;
        let newContent = '';
        for (let info of result.results) {
            let prev = fileContent.substr(curPos, info.start-curPos);
            newContent += prev.replace("\n", "<br>");
            newContent += `<span class="pi-entity" title="${info.entity_type}: ${(info.score*100).toFixed(2)}%">` + fileContent.substr(info.start, info.end+1-info.start) + "</span>";
            curPos = info.end+1;
        }
        if (curPos < fileContent.length)
            newContent += fileContent.substr(curPos)
        getE("detection_output").innerHTML = newContent;

    }
}

async function init() {
    addEvent("btn_exec_test_data", "click", useSynthData);
    addEvent("btn_exec_upload", "click", uploadFile);
}

init();