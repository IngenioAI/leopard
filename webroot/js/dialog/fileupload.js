class FileUploadHanlder extends FileUploader {
    constructor(elementId, progressId) {
        super(elementId);
        this.progressId = progressId;
    }

    setProgress(percent) {
        getE(this.progressId).setAttribute("style", `width:${percent}%`);
    }

    send(url, callback) {
        this.setProgress(0);
        this.callback = callback;
        super.send(url);
    }

    onUploadProgress(event) {
        const progressPercent = (event.loaded / event.total * 100).toFixed();
        this.setProgress(progressPercent);
    }

    onCompleted(event) {
        if (this.callback) {
            this.callback(event, this.getFilenames());
        }
    }
}

class FileUploadDialogBox extends ModalDialogBox {
    constructor(storageId, storagePath) {
        super('LP_DIALOG_fileupload_dialogbox');
        this.storageId = storageId;
        this.storagePath = storagePath;
    }

    onShow(e) {
        getE('LP_DIALOG_fileupload_dialogbox_file').value = "";
        getE('LP_DIALOG_progress').setAttribute("style", `width: 0%`);
        this.addEvent('LP_DIALOG_fileupload_dialogbox_upload', 'click', this.onUpload.bind(this));
    }

    onUpload() {
        if (getE('LP_DIALOG_fileupload_dialogbox_file').value) {
            const uploadHandler = new FileUploadHanlder('LP_DIALOG_fileupload_dialogbox_file', 'LP_DIALOG_progress');
            const url = createStorageFileURL(this.storageId, this.storagePath);
            uploadHandler.send(url, this.onCompleted.bind(this));
        }
        else {
            showMessageBox("파일을 먼저 선택해 주세요", "파일 업로드");
        }
    }

    onCompleted(e, filenames) {
        if (this.resolve) {
            this.resolve(filenames);
        }
        this.hide();
    }
};

function showFileUploadDialogBox(storageId, storagePath, message=null, title=null) {
    const dialogBox = new FileUploadDialogBox(storageId, storagePath);
    dialogBox.setText(message, title);
    return dialogBox.exec();
}