function showMessageBox(message, title) {
    const messageBox = new DialogBox('LP_DIALOG_messagebox');
    messageBox.setText(message, title);
    messageBox.show();
}

class AskMessageBox extends ModalDialogBox {
    constructor(buttons) {
        super('LP_DIALOG_ask_messagebox');
        this.buttons = buttons;
    }

    init() {
        const buttonsDiv = document.getElementById("LP_DIALOG_ask_buttons");
        clearE(buttonsDiv);
        for(let i = 0; i < this.buttons.length; i++) {
            const buttonLabel = this.buttons[i];
            const btn = createE("button", buttonLabel, {
                type: "button",
                class: "btn btn-primary"
            }, {
                click: (e) => this.onClick(i, buttonLabel)
            });
            addE(buttonsDiv, btn);
        }
    }

    onClick(index, buttonLabel) {
        this.close({
            index: index,
            label: buttonLabel
        });
    }
}

function showAskMessageBox(message, title, buttons) {
    const askMessageBox = new AskMessageBox(buttons);
    askMessageBox.setText(message, title);
    return askMessageBox.exec();
}