class FormDialogBox extends ModalDialogBox {
    constructor(formSpec, data, validator) {
        super("LP_DIALOG_form_dialogbox", { dialog_size: "lg"});
        this.formSpec = formSpec;
        this.data = data;
        this.validator = validator;
    }

    init() {
        const formDiv = getE("LP_DIALOG_form_dialogbox_formgroup");
        clearE(formDiv);
        addE(formDiv, createFormGroup(this.formSpec, this.data));
        this.addEvent('LP_DIALOG_form_dialogbox_OK', 'click', this.onOK.bind(this));
    }

    async onOK() {
        const data = getFormGroupData(this.formSpec, this.data);
        let handleOK = true;
        if (this.validator) {
            handleOK = await this.validator(data);
        }
        if (handleOK) {
            this.resolve(data);
            this.hide();
        }
    }
}

function showFormDialogBox(formSpec, data, message, title, validator=null) {
    const formDialogBox = new FormDialogBox(formSpec, data, validator);
    formDialogBox.setText(message, title);
    return formDialogBox.exec();
}