import { createElem, addE, clearE } from "/js/dom_utils.js";
import { getAppList, getExecImageList, getExecList } from "/js/service.js";

function getState(appInfo, execList, imageList) {
    const tagName = appInfo.image.tag;
    for (const execInfo of execList) {
        if (tagName == execInfo.Image) {
            return execInfo.State;
        }
    }
    for (const imageInfo of imageList) {
        if (imageInfo.RepoTags.indexOf(tagName) >= 0) {
            return "ready";
        }
    }
    return "none";
}

function getAppRunLink(appInfo) {
    const linkInfo = [
        ["mtcnn", "face_detect.html"],
        ["faker", "faker.html"],
        ["presidio", "presidio.html"],
        ["facenet", "facenet.html"]
    ];
    for (const info of linkInfo) {
        if (info[0] == appInfo.id) {
            return info[1];
        }
    }
    return "generic_test.html?app_id=" + appInfo.id;
}

async function init() {
    const appTBody = document.getElementById("app_tbody");
    clearE(appTBody);

    const res = await Promise.all([getAppList(), getExecImageList(), getExecList()]);
    //console.log(res)

    const appList = res[0];
    const imageList = res[1];
    const execList = res[2];
    for (const appInfo of appList) {
        const tr = createElem({
            name: "tr", children: [
                { name: "th", attributes: { scope: "row" }, text: appInfo.id },
                { name: "td", text: appInfo.name },
                { name: "td", text: appInfo.type },
                { name: "td", text: appInfo.image.tag },
                { name: "td", text: getState(appInfo, execList, imageList) },
                { name: "td", children: [{ name: "a", attributes: { href: getAppRunLink(appInfo) }, children: [{ name: "span", attributes: { class: "bi bi-arrow-up-right-square" } }] }] }
            ]
        });
        addE(appTBody, tr);
    }
}

init();