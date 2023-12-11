import { ModalDialogBox } from "./dialogbox.js";
import { getE } from "/js/dom_utils.js";
import { createCodeMirrorForPython, createCodeMirrorForJavascript , createCodeMirror } from "/js/codemirror.js";
import { getStorageFileContent } from "/js/service.js";

class CodeEditorDialogBox extends ModalDialogBox {
    constructor(storageId, storagePath) {
        super("code_editor");
        this.storageId = storageId;
        this.storagePath = storagePath;
        this.options.dialog_size = "xl";
        this.editor = null;
        this.content = "";
    }

    async init() {
        const elem = getE("LP_DIALOG_code_editor_textarea");
        if (this.storagePath.endsWith(".py")) {
            this.editor = createCodeMirrorForPython(elem);
        }
        else if (this.storagePath.endsWith(".js")) {
            this.editor = createCodeMirrorForJavascript(elem);
        }
        else if (this.storagePath.endsWith(".json")) {
            this.editor = createCodeMirrorForJavascript(elem, true);
        }
        else {
            this.editor = createCodeMirror(elem);
        }
        const height = window.innerHeight || document.documentElement.clientHeight;
        if (height > 880) {
            this.editor.setSize(null, 600);
        }
        else {
            this.editor.setSize(null, height - 280);
        }
        this.content = await getStorageFileContent(this.storageId, this.storagePath);
    }

    onShow() {
        this.addEvent("LP_DIALOG_code_editor_save", "click", this.onSave.bind(this));
        this.editor.setValue(this.content);
        this.editor.refresh();
    }

    onHidden(e) {
        this.editor.toTextArea();
        super.onHidden(e);
    }

    onSave() {
        this.resolve(this.editor.getValue());
        this.hide();
    }
}

function showCodeEditor(message, title, storageId, storagePath) {
    const dialogBox = new CodeEditorDialogBox(storageId, storagePath);
    dialogBox.setText(message, title);
    return dialogBox.exec();
}

export { CodeEditorDialogBox, showCodeEditor }