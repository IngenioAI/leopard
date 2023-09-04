/*
    DOM utils
        short-cut functions for DOM access
 */

function getE(id) {
    if (typeof id == 'string') {
        return document.getElementById(id);
    }
    else if (id instanceof HTMLElement) {
        return id;
    }
}

function getV(id) {
    const elem = getE(id);
    if ('value' in elem) {
        return elem.value;
    }
    return null;
}

function setV(id, value) {
    getE(id).value = value;
}

function getT(id) {
    return getE(id).innerText;
}

function setT(id, text) {
    getE(id).innerText = text;
}

function createE(tagName, text=null, attributes=null, events=null) {
    const elem = document.createElement(tagName);
    if (text) {
        elem.innerText = text;
    }
    if (attributes) {
        for (let key in attributes) {
            elem.setAttribute(key, attributes[key]);
        }
    }
    if (events) {
        for (let name in events) {
            elem.addEventListener(name, events[name]);
        }
    }
    return elem;
}

function createT(text) {
    return document.createTextNode(text);
}

function addE(parent, child) {
    parent.appendChild(child);
}

function clearE(elem) {
    getE(elem).replaceChildren();
}
