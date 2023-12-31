import { getE, setT } from "/js/dom_utils.js";
import { getSession, deleteSession, getSessionData } from "/js/service.js";
import { createCodeMirrorForPython } from "/js/codemirror.js";

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

async function init() {
    refresh();
    createCodeMirrorForPython(getE("code_test"));

    getE("logout_button").addEventListener("click", logout);
}

init();