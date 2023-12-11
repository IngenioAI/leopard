import { ModalDialogBox } from "./dialogbox.js";
import { getE, clearE, createE, addE } from "/js/dom_utils.js";

class SelectDialogBox extends ModalDialogBox {
    constructor(items) {
        super('select');
        this.items = items;
    }

    init() {
        const selectElem = getE('LP_DIALOG_select_select');
        clearE(selectElem);
        for (const item of this.items) {
            const option = createE("option", item);
            addE(selectElem, option);
        }
    }

    onShow() {
        this.addEvent('LP_DIALOG_select_OK', 'click', this.onOK.bind(this));
        getE('LP_DIALOG_select_select').focus();
    }

    async onOK() {
        const value = getE('LP_DIALOG_select_select').value;
        this.resolve(value);
        this.hide();
    }
}

export function showSelectDialogBox(message, title, items) {
    const dialogBox = new SelectDialogBox(items);
    dialogBox.setText(message, title);
    return dialogBox.exec();
}