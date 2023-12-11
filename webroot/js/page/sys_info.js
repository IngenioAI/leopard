import { setT } from "/js/dom_utils.js";
import { getFileSizeString } from "/js/storage_utils.js";
import { getSysInfo } from "/js/service.js";

async function updateSysInfo() {
    const sysInfo = await getSysInfo();
    setTimeout(updateSysInfo, 1000);
    setT("cpu_info", sysInfo.cpu_info);
    setT("cpu_core_count", sysInfo.cpu_core_count);
    setT("cpu_thread_count", sysInfo.cpu_thread_count);
    setT("cpu_util", sysInfo.cpu_util);
    setT("mem_util", ((1- sysInfo.available_memory / sysInfo.total_memory) * 100).toFixed(2))
    setT("available_memory", getFileSizeString(sysInfo.available_memory));
    setT("total_memory", getFileSizeString(sysInfo.total_memory));
    setT("disk_util", (sysInfo.disk_used / sysInfo.disk_total * 100).toFixed(2))
    setT("available_disk", getFileSizeString(sysInfo.disk_free));
    setT("total_disk", getFileSizeString(sysInfo.disk_total));

    // gpu info
    const gpuInfo = sysInfo.gpu_info;
    const gpu_tbody = document.getElementById('gpu_list_tbody');
    gpu_tbody.innerHTML = '';
    if (gpuInfo) {
        for (let i = 0; i < gpuInfo.length; i++) {
            const table_row = document.createElement("tr");
            let table_data = document.createElement("td");
            table_data.innerText = "GPU-" + i;
            table_row.appendChild(table_data);

            table_data = document.createElement("td");
            table_data.innerText = gpuInfo[i].name;
            table_row.appendChild(table_data);

            table_data = document.createElement("td");
            table_data.innerText = parseInt(gpuInfo[i].gpu_util) + '%';
            table_row.appendChild(table_data);

            table_data = document.createElement("td");
            table_data.innerText = Math.floor(parseInt(gpuInfo[i].used_mem) / parseInt(gpuInfo[i].total_mem) * 100.0) + '%';
            table_row.appendChild(table_data);

            table_data = document.createElement("td");
            table_data.innerText = gpuInfo[i].temp;
            table_row.appendChild(table_data);

            gpu_tbody.appendChild(table_row);
        }
    }

}

async function init() {
    setTimeout(updateSysInfo, 0);
}

init();