class LogViewDialogBox extends DialogBox {
    constructor() {
        super('LP_DIALOG_logview');
    }

    async init() {
        const contentDiv = getE('LP_DIALOG_logview_content');
        clearE(contentDiv);
    }

    setLog(text) {
        setT('LP_DIALOG_logview_content', text)
    }
}

function showLogView(message, title) {
    const dialogBox = new LogViewDialogBox();
    dialogBox.setText(message, title);
    dialogBox.show();
    return dialogBox;
}