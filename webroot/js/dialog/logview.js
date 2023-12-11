import { DialogBox } from "./dialogbox.js";
import { getE, clearE } from "/js/dom_utils.js";

const Terminal = window.Terminal;

class LogViewDialogBox extends DialogBox {
    constructor() {
        super('LP_DIALOG_logview');
        this.log = '';
        this.term = null;
    }

    async init() {
        const contentDiv = getE('LP_DIALOG_logview_content');
        clearE(contentDiv);
        this.log = '';

        this.term = new Terminal({disableStdin: true, convertEol: true});
        this.term.open(getE("LP_DIALOG_logview_content"));
        this.term.resize(80, 25);
    }

    clearLog() {
        this.log = '';
    }

    setLog(text) {
        this.term.write(text.substr(this.log.length));
        this.log = text;
    }

    addLog(text) {
        this.term.write(text);
        this.log += text;
    }
}

export function showLogView(message, title) {
    const dialogBox = new LogViewDialogBox();
    dialogBox.setText(message, title);
    dialogBox.show();
    return dialogBox;
}