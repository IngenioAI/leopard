import { getE, setT } from "/js/dom_utils.js";
import { getSession, deleteSession, getSessionData } from "/js/service.js";
import { createCodeMirrorForPython } from "/js/codemirror.js";
import { showAskMessageBox } from "/js/dialog/ask_messagebox.js";

async function logout() {
    await deleteSession();
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

async function dlg_test() {
    const ans1 = await showAskMessageBox("테스트1", "테스트", ["1", "2"])
    console.log(ans1);
    const ans2 = await showAskMessageBox("테스트3", "테스트", ["3", "4"])
    console.log(ans2);
}

async function init() {
    refresh();
    createCodeMirrorForPython(getE("code_test"));

    getE("logout_button").addEventListener("click", logout);
    getE("dlg_test").addEventListener("click", dlg_test);
}

init();