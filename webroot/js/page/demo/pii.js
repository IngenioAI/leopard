import { getE, getV, addEvent, showE, hideE } from "/js/dom_utils.js";
import { joinPath } from "/js/storage_utils.js";
import { runApp, getStorageFileContent } from "/js/service.js";

import { CSV } from "/js/csv.js";
import { csvToGrid } from "/js/csv_grid.js";

import { showMessageBox } from "/js/dialog/messagebox.js";
import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";

async function useSynthData() {
    hideE("detection_output_div");
    hideE("csv_grid_div");
    showE("generate_params");
}

async function generateRun() {
    const result = await runApp('pii', {
        module: "presidio",
        action_type : 'generate_and_analyze',
        generate_info: {
            base: getV("generate_base"),
            field: [getV("generate_field")],
            count: parseInt(getV("generate_count")),
        }
    });

    // sort result
    result.results.sort((a, b) => {
        return a.start - b.start
    });

    hideE("detection_output_div");
    showE("csv_grid_div");

    let fileContent = result.text_content;
    drawCsvGrid(result.results, fileContent);
}

function drawCsvGrid(results, fileContent) {
    const ret = CSV.parse(fileContent, '|', true);
    const decoCells = [];
    for (let info of results) {
        for (let row = 0; row < ret.cellInfo.length; row++) {
            for (let column = 0; column < ret.cellInfo[row].length; column++) {
                const cell = ret.cellInfo[row][column];
                if (cell[0] <= info.start && cell[1] >= info.end) {
                    let cellColor = "#FFFF77";
                    if (info.entity_type == "주민등록번호") {
                        cellColor = "#FF7777";
                    }
                    for (let decoCell of decoCells) {
                        if (decoCell.column == column && decoCell.row == row) {
                            if (info.entity_type == "주민등록번호") {
                                decoCell.style = { backgroundColor: cellColor};
                                decoCell.toolTip = `${info.entity_type}: ${(info.score*100).toFixed(1)}%`;
                            }
                            continue;
                        }
                    }
                    decoCells.push({
                        column: column,
                        row: row,
                        style: { backgroundColor: cellColor },
                        toolTip: `${info.entity_type}: ${(info.score*100).toFixed(1)}%`,
                    });
                }
            }
        }
    }
    csvToGrid("csv_grid", ret.data, decoCells);

}
async function uploadFile() {
    hideE("generate_params");
    const res = await showFileUploadDialogBox('0', 'app/pii/input');
    if (res.success) {
        showE("detection_output_div");
        hideE("csv_grid_div");
        getE("detection_output").innerHTML = "처리중... 잠시만 기다려 주십시오.";

        const fileName = res.files[0];
        const result = await runApp('pii', {
            module: "presidio",
            action_type : 'analyze',
            text_file: fileName
        });

        // sort result
        result.results.sort((a, b) => {
            return a.start - b.start
        });

        let fileContent = await getStorageFileContent("0", joinPath("app/pii/input", fileName));
        fileContent = fileContent.replaceAll("\r", "");
        if (fileName.endsWith(".csv")) {
            hideE("detection_output_div");
            showE("csv_grid_div");
            drawCsvGrid(result.results, fileContent);
        }
        else {
            let curPos = 0;
            let newContent = '';
            for (let info of result.results) {
                let prev = fileContent.substr(curPos, info.start-curPos);
                let entity_str = fileContent.substr(info.start, info.end-info.start);
                newContent += prev.replaceAll("\n", "<br>");
                newContent += `<span class="pi-entity" data-bs-toggle="tooltip" data-bs-title="${info.entity_type}: ${(info.score*100).toFixed(2)}%">${entity_str}</span>`;
                curPos = info.end;
            }
            if (curPos < fileContent.length) {
                newContent += fileContent.substr(curPos).replaceAll("\n", "<br>")
            }
            getE("detection_output").innerHTML = newContent;

            const bootstrap = window.bootstrap;
            const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]')
            const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl))
        }
    }
}

async function init() {
    addEvent("btn_exec_test_data", "click", useSynthData);
    addEvent("btn_exec_upload", "click", uploadFile);
    addEvent("btn_exec_generate_run", "click", generateRun);
}

init();