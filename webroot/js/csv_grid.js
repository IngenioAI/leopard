import { CSV } from "/js/csv.js";

const agGrid = window.agGrid;

export function csvToGrid(gridId, csvData, decoCells=null) {
    let csvLines;
    if (Array.isArray(csvData)) {
        csvLines = csvData;
    }
    else {
        csvLines = CSV.parse(csvData);
    }

    const headers = csvLines[0];
    const columnDefs = [];
    for (const column of headers) {
        columnDefs.push({
            field: column,
            cellStyle: p => {
                if (decoCells) {
                    const columnIndex = headers.indexOf(p.colDef.field);
                    for (let info of decoCells) {
                        if (info.column == columnIndex && info.row == p.rowIndex+1) {
                            return info.style;
                        }
                    }
                }
                return null;
            },
            tooltipValueGetter: (p) => {
                if (decoCells) {
                    const columnIndex = headers.indexOf(p.colDef.field);
                    for (let info of decoCells) {
                        if (info.column == columnIndex && info.row == p.rowIndex+1) {
                            return info.toolTip;
                        }
                    }
                }
                return null;
            },
            resizable: true
        });
    }
    const rowData = [];
    for (let i = 1; i < csvLines.length; i++) {
        if (csvLines[i].length == headers.length) {
            const row = {};
            for (let j = 0; j < headers.length; j++) {
                row[headers[j]] = csvLines[i][j];
            }
            rowData.push(row)
        }
    }
    const gridOptions = {
        columnDefs: columnDefs,
        rowData: rowData,
        tooltipShowDelay: 1000,
    };
    const gridDiv = document.getElementById(gridId);
    gridDiv.innerHTML = "";
    //return new agGrid.Grid(gridDiv, gridOptions);     // old version
    return agGrid.createGrid(gridDiv, gridOptions);
}
