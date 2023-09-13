class InputDialogBox extends ModalDialogBox {
    constructor(validator) {
        super('LP_DIALOG_input_dialogbox');
        this.validator = validator;
    }

    onShow(e) {
        getE('LP_DIALOG_input_dialogbox_input').value = "";
        this.addEvent('LP_DIALOG_input_dialogbox_OK', 'click', this.onOK.bind(this));
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
};

function showInputDialogBox(message, title, validator=null) {
    const dialogBox = new InputDialogBox(validator);
    dialogBox.setText(message, title);
    return dialogBox.exec();
}