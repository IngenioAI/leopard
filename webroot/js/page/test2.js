import { getSession, getSessionData } from "/js/service.js";

async function init() {
    const session = await getSession();
    console.log(session);
    const sessionData = await getSessionData();
    console.log(sessionData);
}

init();