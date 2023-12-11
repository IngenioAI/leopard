import { csvToGrid } from "/js/csv_grid.js";
import { getE, getV } from "/js/dom_utils.js";
import { joinPath, downloadStorageFile } from "/js/storage_utils.js";
import { runApp, getStorageFileContent, uploadFileToStorage } from "/js/service.js";
import { showFileSave } from "/js/dialog/filesave.js";

let csvData = null;
let csvFilePath = "";

async function generateData(type) {
    // clear previous data
    document.getElementById("view_csv").innerText = "";
    document.getElementById("view_csv").style = "display:none";
    csvToGrid("myGrid", "");

    const dataCount = parseInt(getV('input_count'));
    const res = await runApp("faker", {
        type: type,
        count: dataCount
    });
    csvFilePath = res.text_path;
    csvData = await getStorageFileContent('0', joinPath('/app/faker/run', csvFilePath));
    csvToGrid("myGrid", csvData);
}
function showRawData() {
    document.getElementById("view_csv").innerText = csvData;
    document.getElementById("view_csv").style = "display:block; max-height:600px";
}
function downloadCsv() {
    downloadStorageFile('0', joinPath("/app/faker/run", csvFilePath), "faker_data.csv");
}
async function saveCsv() {
    const filePath = await showFileSave({ defaultFilename: 'faker_data.csv' });
    const storagePath = filePath.split(":");
    await uploadFileToStorage(storagePath[0], storagePath[1], csvData);
}

async function init() {
    getE("generate_personal_data_button").addEventListener("click", ()=>generateData("personal"));
    getE("generate_log_data_button").addEventListener("click", ()=>generateData("log"));
    getE("show_raw_data_button").addEventListener("click", showRawData);
    getE("download_csv_button").addEventListener("click", downloadCsv);
    getE("save_csv_button").addEventListener("click", saveCsv);
}

init();