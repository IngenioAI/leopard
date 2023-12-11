import { ModalDialogBox  } from "./dialogbox.js";

export function showMessageBox(message, title) {
    const messageBox = new ModalDialogBox("messagebox");
    messageBox.setText(message, title);
    return messageBox.exec();
}