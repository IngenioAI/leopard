import { createElem } from "/js/dom_utils.js"
const bootstrap = window.bootstrap;

function createTab(tabInfo, defaultTab=null, onShow=null, onHide=null) {
    if (defaultTab == null) {
        defaultTab = tabInfo[0].id;
    }
    const tab = { name: "ul", attributes: { class: "nav nav-tabs" }, children:
        tabInfo.map((item) => {
            return { name: "li", attributes: { class: "nav-item"}, children: [
                { name: "button", text: item.text, attributes: { class: "nav-link" + ((item.id==defaultTab) ? " active" : ""), id: `tab_${item.id}`, 'data-bs-target': `#${item.id}`, 'data-bs-toggle': "tab"}}
            ]}
        })
    };
    const elem = createElem(tab);
    if (onShow) {
        elem.addEventListener("show.bs.tab", onShow);
    }
    if (onHide) {
        elem.addEventListener("hide.bs.tab", onHide);
    }
    return elem;
}

function showTab(tabId) {
    let buttonOutput = document.getElementById(`tab_${tabId}`);
    if (buttonOutput == null) {
        buttonOutput = document.getElementById(tabId);
    }
    if (buttonOutput) {
        let tabOutput = bootstrap.Tab.getInstance(buttonOutput)
        if (tabOutput == null)
        tabOutput = new bootstrap.Tab(buttonOutput)
        tabOutput.show();
    }
    else {
        console.warn("ID not found for Tab:", tabId);
    }
}

export { createTab, showTab }