let sourceUploadInfo = null;
let currentLog = null;
let logWindow = null;

function createExecItem(itemInfo, execInfo) {
    return createListGroupItem(
        [
            {
                name: "h5", attributes: { class: "mb-1" }, children: [
                    { name: "span", attributes: { class: "", style: "padding-right: 0.2em" } },
                    { name: "span", text: itemInfo.id }
                ]
            },
            {
                name: "div", attributes: { class: "d-flex w-100 justify-content-between" }, children: [
                    { name: "small", text: itemInfo.command },
                    { name: "small", text: itemInfo.imageTag }
                ]
            },
            { name: "div", text: "running", attributes: { id: `state_${itemInfo.id}` } }
        ],
        (e) => {
            logWindow = showLogView(`${itemInfo.command} on ${itemInfo.imageTag}`, `실행 로그 - ${itemInfo.id}`);
            if (currentLog) {
                logWindow.setLog(currentLog);
            }
        });
}

async function createExecution() {
    const item = createPostItem([
        ["id", "input_exp_id"],
        ['srcPath', 'src_path'],
        ['command', 'command_line'],
        ['imageTag', 'select_images'],
        ['inputPath', 'input_data_path'],
        ['outputPath', 'output_data_path']
    ]);
    if (item["id"] == "") {
        showMessageBox("실험 ID를 입력해야 합니다.", "실행");
        return;
    }
    if (sourceUploadInfo && sourceUploadInfo.success) {
        item['uploadId'] = sourceUploadInfo.upload_id;
    }
    const execInfo = await createExec(item);
    execInfo.id = item['id'];
    setTimeout(checkLogs, 1000, execInfo);

    const elem = getE("exec_list");
    addE(elem, createExecItem(item, execInfo));
    showTab("current_exec");
}

async function checkLogs(info) {
    const inspectInfo = await inspectExec(info.exec_id);
    const logs = await getExecLogs(info.exec_id);
    if (inspectInfo.State.Running) {
        setTimeout(checkLogs, 1000, info);
        setT(`state_${info.id}`, "running");
    }
    else {
        removeExec(info.exec_id);    // auto remove after getting logs
        setT(`state_${info.id}`, "exited");
    }
    currentLog = logs.lines;
    if (logWindow) {
        logWindow.setLog(currentLog);
    }
}

async function setInputPath() {
    const filepath = await showFileSave('1', '');
    console.log(filepath)
}

async function setSourceCode() {
    const res = await showFileUploadDialogBox();
    if (sourceUploadInfo) {
        if (sourceUploadInfo.success) {
            removeUploadItem(sourceUploadInfo.upload_id);
        }
    }
    sourceUploadInfo = res;
    setV("src_path", sourceUploadInfo.files[0]);
}

async function setCommandLine() {
    const command = await showInputDialogBox("커맨드 라인 입력", "입력");
    setV("command_line", command);
}

async function refreshImageList() {
    const imageList = await getExecImageList();

    const selectImages = document.getElementById("select_images");
    while (selectImages.length > 0) {
        selectImages.remove(0);
    }

    for (const im of imageList) {
        const option = document.createElement("option");
        option.text = im.RepoTags;
        selectImages.add(option);
    }
}

async function init() {
    const tab = createTab([
        { id: "new_exec", text: "새 실험/실행" },
        { id: "current_exec", text: "실행 중" },
        { id: "completed_exec", text: "실행 완료" }]);
    addE("tab_div", tab);

    refreshImageList();

    window.addEventListener("beforeunload", (e) => {
        if (sourceUploadInfo) {
            removeUploadItem(sourceUploadInfo.upload_id);
            console.log('remove:', sourceUploadInfo.upload_id)
        }
    });
}