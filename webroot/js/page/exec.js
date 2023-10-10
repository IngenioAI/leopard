let sourceUploadInfo = null;
let currentLogId = null;
let logWindow = null;

function createExecItem(execInfo, initState="실행 중") {
    const MENU_ID = {
        STOP: 100,
        DELETE: 101,
        VIEW: 102,
        RESTART: 103
    };
    const contextHandler = async (menuId, info) => {
        if (menuId == MENU_ID.STOP) {
            const res = await stopExec(info.id);
            if (!res.success) {
                showMessageBox(res.error_message, "실행 중지 오류");
            }
            refreshExecList();
        }
        else if (menuId == MENU_ID.DELETE) {
            const res = await removeExec(info.id);
            if (!res.success) {
                showMessageBox(res.error_message, "실행 삭제 오류");
            }
            refreshExecList();
        }
    }
    const contextMenuExited = new ContextMenu([
        { id: MENU_ID.DELETE, title: "삭제", info: execInfo },
        { id: MENU_ID.VIEW, title: "세부 정보", info: execInfo },
        { id: MENU_ID.RESTART, title: "다시 실행", info: execInfo }
    ], contextHandler);
    const contextMenuRunning = new ContextMenu([
        { id: MENU_ID.STOP, title: "중지", info: execInfo },
        { id: MENU_ID.VIEW, title: "세부 정보", info: execInfo }
    ], contextHandler);
    return createListGroupItem([
            { name: "h5", attributes: { class: "mb-1" }, children: [
                { name: "span", attributes: { class: "", style: "padding-right: 0.2em" } },
                { name: "span", text: execInfo.id }]
            },
            { name: "div", attributes: { class: "d-flex w-100 justify-content-between" }, children: [
                { name: "small", text: execInfo.command_line },
                { name: "small", text: execInfo.base_image }]
            },
            { name: "div", text: initState, attributes: { id: `state_${execInfo.id}` } }
        ],
        (e) => {    // listItemClick
            logWindow = showLogView(`${execInfo.command_line} on ${execInfo.base_image}`, "실행 로그");
            logWindow.clearLog();
            currentLogId = execInfo.id;
            checkLogs(execInfo);
        },
        initState == "실행 중" ? contextMenuRunning.element : contextMenuExited.element);
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
        await showMessageBox("실험 ID를 입력해야 합니다.", "실행");
        getE("input_exp_id").focus();
        return;
    }
    if (sourceUploadInfo && sourceUploadInfo.success) {
        item['uploadId'] = sourceUploadInfo.upload_id;
    }
    const res = await createExec(item);
    if (res.success) {
        const execInfo = res.exec_info;
        refreshExecList();
        showTab("current_exec");
        checkLogs(execInfo);
    }
    else {
        showMessageBox(res.error_message, "실행 오류");
    }
}

async function checkLogs(info) {
    if (info.id == currentLogId) {
        const execInfo = await getExecInfo(info.id);
        if (info.id == currentLogId) {
            if ('State' in execInfo.container && execInfo.container.State.Running) {
                setTimeout(checkLogs, 1000, info);
                setT(`state_${info.id}`, "실행 중");
            }
            else {
                setT(`state_${info.id}`, "종료됨");
            }

            if (logWindow) {
                const logs = await getExecLogs(info.id);
                if (info.id == currentLogId) {
                    logWindow.setLog(logs.lines);
                }
            }
        }
    }
}

async function setInputPath() {
    const filepath = await showSelectPath();
    setV("input_data_path", filepath);
}

async function setOutputPath() {
    const filepath = await showSelectPath();
    setV("output_data_path", filepath);
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
    if (sourceUploadInfo.files[0].indexOf(".py") > 0) {
        setV("command_line", `python ${sourceUploadInfo.files[0]}`)
    }
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

async function refreshExecList() {
    const execList = await getExecList();
    const currentList = getE("exec_current_list");
    clearE(currentList)
    const completedList = getE("exec_completed_list");
    clearE(completedList)
    for (const execInfo of execList) {
        if ('State' in execInfo.container && execInfo.container.State.Running) {
            addE(currentList, createExecItem(execInfo, "실행 중"));
        }
        else {
            addE(completedList, createExecItem(execInfo, "종료됨"));
        }
    }
}

async function init() {
    const tab = createTab([
        { id: "new_exec", text: "새 실험/실행" },
        { id: "current_exec", text: "실행 중" },
        { id: "completed_exec", text: "실행 완료" }],
        null,
        (event) => {
            //console.log(event.target.id);
            refreshExecList();
        });
    addE("tab_div", tab);

    refreshImageList();
    refreshExecList();

    window.addEventListener("beforeunload", (e) => {
        if (sourceUploadInfo) {
            removeUploadItem(sourceUploadInfo.upload_id);
            console.log('remove:', sourceUploadInfo.upload_id)
        }
    });
}