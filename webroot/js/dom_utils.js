/*
    DOM utils
        short-cut functions for DOM access
 */

export function getE(id) {
    if (typeof id == 'string') {
        return document.getElementById(id);
    }
    else if (id instanceof HTMLElement) {
        return id;
    }
}

export function getV(id) {
    const elem = getE(id);
    if ('value' in elem) {
        return elem.value;
    }
    return null;
}

export function setV(id, value) {
    getE(id).value = value;
}

export function getT(id) {
    return getE(id).innerText;
}

export function setT(id, text) {
    getE(id).innerText = text;
}

export function createE(tagName, text=null, attributes=null, events=null) {
    const elem = document.createElement(tagName);
    if (text) {
        elem.innerText = text;
    }
    if (attributes) {
        for (let key in attributes) {
            if (attributes[key] != null)
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

export function createT(text) {
    return document.createTextNode(text);
}

export function addE(parent, child) {
    const parentElem = getE(parent);
    parentElem.appendChild(child);
    return parentElem;
}

export function clearE(elem) {
    const e =getE(elem);
    e.replaceChildren();
    return e;
}

export function createElem(o) {
    const elem = createE(o.name, o.text, o.attributes, o.events);
    if ('children' in o) {
        for (const child of o.children) {
            const childElem = createElem(child);
            addE(elem, childElem);
        }
    }
    return elem;
}

export function elementToJson(id) {
    const element = getE(id);
    const o = {
        name: element.nodeName.toLowerCase()
    };
    if (element.attributes != null) {
        if (element.attributes.length) {
            o["attributes"] = {};
            for (let i = 0; i < element.attributes.length; i++) {
                o["attributes"][element.attributes[i].nodeName] = element.attributes[i].nodeValue;
            }
        }
    }
    o['children'] = [];
    var nodeList = element.childNodes;
    if (nodeList != null) {
        if (nodeList.length) {
            for (let i = 0; i < nodeList.length; i++) {
                if (nodeList[i].nodeType == 3) {
                    o['text'] = nodeList[i].nodeValue.trim();
                } else {
                    const child = elementToJson(nodeList[i]);
                    o['children'].push(child);
                }
            }
        }
    }
    return o;
}

export function elementToString(id) {
    const element = getE(id);
    return element.outerHTML;
}

export function isJSONEmpty(object) {
    return JSON.stringify(object) === "{}";
}
