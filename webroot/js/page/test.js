import { getE, setT } from "/js/dom_utils.js";
import { createSession, getSession, deleteSession } from "/js/service.js";
import { createCodeMirrorForPython } from "/js/codemirror.js";
import { showMessageBox } from "/js/dialog/messagebox.js";
import { showAskMessageBox } from "/js/dialog/ask_messagebox.js";

async function login() {
    const res = await createSession(getV("input_user"));
    await saveSessionData({"test": "test string"})
    refresh();
}
async function logout() {
    const res = await deleteSession();
    refresh();
}
async function refresh() {
    const session = await getSession();
    if (session.success) {
        setT("username", session.username);
        const data = await getSessionData();
        console.log(data);
    }
    else {
        setT("username", "사용자 없음");
    }

}

function testgo() {
    showAskMessageBox("abc", "def", ["YES", "NO", "CANCEL"]);
}

async function init() {
    refresh();
    createCodeMirrorForPython(getE("code_test"));

    getE("test_button").addEventListener("click", testgo);
}

init();