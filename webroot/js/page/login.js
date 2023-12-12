import { getE, getV, setT } from "/js/dom_utils.js";
import { createSession, saveSessionData, deleteSession, getSession, getSessionData } from "/js/service.js";

async function login() {
    await createSession(getV("input_user"));
    await saveSessionData({ "test": "test string" })
    refresh();
}

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

    getE("login_button").addEventListener("click", login);
    getE("logout_button").addEventListener("click", logout);
}

init();