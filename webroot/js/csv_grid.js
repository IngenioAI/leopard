import { CSV } from "./csv.js";

const agGrid = window.agGrid;

export function csvToGrid(gridId, csvData) {
    const csvLines = CSV.parse(csvData)
    const headers = csvLines[0];
    const columnDefs = [];
    for (const column of headers) {
        columnDefs.push({ field: column, resizable: true })
    }
    const rowData = [];
    for (let i = 1; i < csvLines.length; i++) {
        const row = {};
        for (let j = 0; j < headers.length; j++) {
            row[headers[j]] = csvLines[i][j];
        }
        rowData.push(row)
    }
    const gridOptions = {
        columnDefs: columnDefs,
        rowData: rowData
    };
    const gridDiv = document.getElementById(gridId);
    gridDiv.innerHTML = "";
    return new agGrid.Grid(gridDiv, gridOptions);
}
