import { createElem } from "/js/dom_utils.js"

function createListGroupItem(child, onClick=null, contextMenu=null) {
    let children = [];
    let isHTMLElement = false;
    if (child instanceof Array) {
        if (child[0] instanceof HTMLElement) {
            children = child;
            isHTMLElement = true;
        }
        else {
            children = child;
        }
    }
    else {
        if (child[0] instanceof HTMLElement) {
            children = [child];
            isHTMLElement = true;
        }
        else {
            children = [child];
        }
    }
    const itemElem = createElem({
        name: "div", attributes: { class: "list-group-item list-group-item-action", 'aria-current': "true" }, children: [
            { name: "div", attributes: { class: "d-flex w-100 justify-content-between" }, children: [
                { name: "div", attributes: { class: "w-100" }, events: { click: (e) => onClick(e) },
                    children: isHTMLElement ? [] : children }
            ]}
        ]
    });
    if (isHTMLElement) {
        for (const elem of children) {
            itemElem.firstChild.firstChild.appendChild(elem);
        }
    }
    if (contextMenu) {
        itemElem.firstChild.appendChild(contextMenu);
    }
    return itemElem;
}

export { createListGroupItem }