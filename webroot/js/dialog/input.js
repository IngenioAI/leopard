class InputDialogBox extends ModalDialogBox {
    constructor(validator) {
        super('LP_DIALOG_input_dialogbox');
        this.validator = validator;
    }

    init() {
        getE('LP_DIALOG_input_dialogbox_input').value = "";
    }

    onShow(e) {
        this.addEvent('LP_DIALOG_input_dialogbox_OK', 'click', this.onOK.bind(this));
        this.addEvent('LP_DIALOG_input_dialogbox_input', 'keypress', this.onKeyPress.bind(this));
        getE('LP_DIALOG_input_dialogbox_input').focus();
    }

    async onOK() {
        const value = getE('LP_DIALOG_input_dialogbox_input').value;
        let handleOK = true;
        if (this.validator) {
            handleOK = await this.validator(value);
        }
        if (handleOK) {
            this.resolve(value);
            this.hide();
        }
    }

    async onKeyPress(e) {
        if (e.key == "Enter") {
            this.onOK();
        }
    }
};

function showInputDialogBox(message, title, validator=null) {
    const dialogBox = new InputDialogBox(validator);
    dialogBox.setText(message, title);
    return dialogBox.exec();
}