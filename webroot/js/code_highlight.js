import { createE, addE } from "/js/dom_utils.js";

const hljs = window.hljs;

export function codeHighlight(elem, content, filepath=null) {
    let highlightedContent = content;
    if (filepath.endsWith(".py")) {
        highlightedContent = hljs.highlight(content, {language: 'python'}).value;
    }
    else {
        highlightedContent = hljs.highlightAuto(content).value;
    }
    const codeDiv = createE("pre", "", { class: "border p-2 bg-light" })
    codeDiv.innerHTML = highlightedContent;
    addE(elem, codeDiv);
}
