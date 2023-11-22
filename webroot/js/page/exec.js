let sourceUploadInfo = null;
let currentLogId = null;
let logWindow = null;
let progressMonitoring = false;

const execUserdata = {
    dataset: "",
    model: ""
}

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
    item["userdata"] = execUserdata;
    const res = await createExec(item);
    if (res) {
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

async function checkProgress() {
    if (progressMonitoring) {
        let runningCount = 0;
        const execList = await getExecList();
        for (const execInfo of execList) {
            if ('State' in execInfo.container && execInfo.container.State.Running) {
                const progressInfo = await getExecProgress(execInfo.id);
                //console.log(execInfo, progressInfo, new Date().toLocaleString());
                if (progressInfo) {
                    setT(`state_${execInfo.id}`, `실행 중 - 진행 ${progressInfo.main_progress.toFixed(1)}%`);
                }
                else {
                    setT(`state_${execInfo.id}`, `실행 중 - 진행 준비`);
                }
                runningCount += 1;
            }
            else {
                setT(`state_${execInfo.id}`, `종료됨`);
            }
        }

        if (runningCount <= 0) {
            progressMonitoring = false;
        }
    }
}

async function setInputPath() {
    const filepath = await showSelectPath();
    if (filepath) {
        setV("input_data_path", filepath);
        execUserdata.dataset = "";
    }
}

async function setOutputPath() {
    const filepath = await showSelectPath();
    if (filepath) {
        setV("output_data_path", filepath);
    }
}

function clearUploadInfo() {
    if (sourceUploadInfo) {
        if (sourceUploadInfo.success) {
            removeUploadItem(sourceUploadInfo.upload_id);
        }
    }
    sourceUploadInfo = null;
}

async function setSourceCode() {
    const res = await showFileUploadDialogBox(null, "/", "업로드할 소스 코드를 선택하세요", "소스 코드 업로드");
    if (res) {
        clearUploadInfo();
        sourceUploadInfo = res;
        setV("src_path", sourceUploadInfo.files[0]);
        execUserdata.model = "";
        if (sourceUploadInfo.files[0].indexOf(".py") > 0) {
            setV("command_line", `python ${sourceUploadInfo.files[0]}`)
        }
    }
}

async function setCommandLine() {
    const command = await showInputDialogBox("커맨드 라인 입력", "입력");
    if (command) {
        setV("command_line", command);
        execUserdata.model = "";
    }
}

async function setInputDataset() {
    const datasetList = await getDatasetList();

    const res = await showSelectDialogBox("입력 데이터로 사용할 데이터셋을 선택하세요", "데이터셋 설정",
        datasetList.map((item) => item.name));
    if (res) {
        const dataset = datasetList.find((item) => item.name == res);
        setV("input_data_path", `${dataset.storageId}:${dataset.storagePath}`);
        execUserdata.dataset = dataset.name;
    }
}

async function setModel() {
    const modelList = await getModelList();

    const res = await showSelectDialogBox("실행할 코드 모델을 선택하세요", "소스 설정",
        modelList.map((item) => item.name));
    if (res) {
        const model = modelList.find((item) => item.name == res);
        clearUploadInfo();
        setV("src_path", `${model.storageId}:${model.storagePath}`);
        setV("command_line", `python ${model.mainSrc}`);
        execUserdata.model = model.name;
    }
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
    let runningCount = 0;
    const execList = await getExecList();
    const currentList = getE("exec_current_list");
    clearE(currentList)
    const completedList = getE("exec_completed_list");
    clearE(completedList)
    for (const execInfo of execList) {
        if ('State' in execInfo.container && execInfo.container.State.Running) {
            addE(currentList, createExecItem(execInfo, "실행 중"));
            runningCount += 1;
        }
        else {
            addE(completedList, createExecItem(execInfo, "종료됨"));
        }
    }

    if (runningCount > 0) {
        progressMonitoring = true;
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

    setInterval(checkProgress, 1000);

    window.addEventListener("beforeunload", (e) => {
        if (sourceUploadInfo) {
            removeUploadItem(sourceUploadInfo.upload_id);
            console.log('remove:', sourceUploadInfo.upload_id)
        }
    });
}