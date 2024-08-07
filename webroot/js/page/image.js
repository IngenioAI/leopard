import { getE, addE, clearE, createE, createElem } from "/js/dom_utils.js";
import { createExecImage, getExecImageCreationInfo, removeExecImageCreationInfo, removeExecImage, getExecImageList } from "/js/service.js";
import { getElapsedTimeString, getFileSizeString } from "/js/storage_utils.js";

import { showFormDialogBox } from "/js/dialog/formdialog.js";
import { showLogView } from "/js/dialog/logview.js";
import { showMessageBox } from "/js/dialog/messagebox.js";
import { ContextMenu } from "/js/control/context_menu.js";

let logView;

async function onClickLeopardOnly() {
    refreshImageTable();
}

async function checkBuild(name) {
    const createInfo = await getExecImageCreationInfo(name);
    if (createInfo.status != 'exited') {
        setTimeout(checkBuild, 500, name);
    }
    else {
        await removeExecImageCreationInfo(name);
        refreshImageTable();
    }
    logView.setLog(createInfo.lines);
}

async function onCreateImage() {
    const data = await showFormDialogBox([
        { id: "name", title: "이름", type: "text", default: "leopard/" },
        { id: "baseImage", title: "기본 이미지", type: "text", default: "python:3.8" },
        { id: "update", title: "업데이트", type: "bool", default: true },
        { id: "aptInstall", title: "apt 설치", type: "text" },
        { id: "pipInstall", title: "pip 설치", type: "text" },
        { id: "additionalCommand", title: "추가 명령 실행", type: "text" }
    ], null, "다음 값으로 이미지를 생성합니다", "이미지 생성");
    if (data) {
        //data['update'] = true;

        const result = await createExecImage(data);
        if (result.success) {
            logView = showLogView(`${data['name']} 이미지 생성`, "이미지 생성 로그");
            setTimeout(checkBuild, 100, data['name']);
        }
    }
}

function createContextMenu(imageInfo) {
    const MENU_ID = {
        DELETE: 100,
        VIEW: 101,
        EDIT: 102,
        BUILD: 103
    };
    const contextMenu = new ContextMenu([
            { id: MENU_ID.DELETE, title: "삭제", info: imageInfo },
            { id: MENU_ID.VIEW, title: "세부 정보", info: imageInfo }
        ],
        async (menuId, info) => {
            if (menuId == MENU_ID.DELETE) {
                const res = await removeExecImage(info.RepoTags[0]);
                if (res.success) {
                    refreshImageTable();
                }
                else {
                    showMessageBox(res.error_message, "에러");
                }
            }
        });
    return contextMenu.element;
}

async function refreshImageTable() {
    const imageList = await getExecImageList();
    //console.log(imageList)

    clearE("image_tbody");
    for (const imageInfo of imageList) {
        if (!getE("leopard_only_checkbox").checked || imageInfo.RepoTags[0].startsWith("leopard/")) {
            const tr = createElem({
                name: "tr", children: [
                    {
                        name: "td", children: [{
                            name: "a", text: imageInfo.RepoTags[0], attributes: { href: "#", class: "text-decoration-none" },
                            events: {
                                click: () => {
                                    console.log("click", imageInfo.RepoTags[0]);
                                }
                            }
                        }]
                    },
                    { name: "td", text: imageInfo.Id.substr(7, 12) },
                    { name: "td", text: getElapsedTimeString(new Date(imageInfo.Created * 1000)) },
                    { name: "td", text: getFileSizeString(imageInfo.Size) },
                ]
            });
            const lastTD = createE("td", "", { class: "align-top" });
            addE(lastTD, createContextMenu(imageInfo));
            addE(tr, lastTD);
            addE("image_tbody", tr);
        }
    }
}

async function init() {
    refreshImageTable();
    getE("leopard_only_checkbox").addEventListener("click", onClickLeopardOnly);
    getE("create_image_button").addEventListener("click", onCreateImage);
}

init();