class FileUploadHanlder extends FileUploader {
    constructor(elementId, progressId) {
        super(elementId);
        this.progressId = progressId;
    }

    setProgress(percent) {
        getE(this.progressId).setAttribute("style", `width:${percent}%`);
    }

    send(url, callback, metadata=null) {
        this.setProgress(0);
        this.callback = callback;
        super.send(url, metadata);
    }

    onUploadProgress(event) {
        const progressPercent = (event.loaded / event.total * 100).toFixed();
        this.setProgress(progressPercent);
    }

    onCompleted(event) {
        if (this.callback) {
            this.callback(event, JSON.parse(this.response));
        }
    }
}

class FileUploadDialogBox extends ModalDialogBox {
    constructor(storageId=null, storagePath="/") {
        super("fileupload");
        this.storageId = storageId;
        this.storagePath = storagePath;
    }

    onShow(e) {
        getE('LP_DIALOG_fileupload_file').value = "";
        getE('LP_DIALOG_fileupload_progress').setAttribute("style", `width: 0%`);
        this.addEvent('LP_DIALOG_fileupload_upload', 'click', this.onUpload.bind(this));
    }

    onUpload() {
        if (getE('LP_DIALOG_fileupload_file').value) {
            const uploadHandler = new FileUploadHanlder('LP_DIALOG_fileupload_file', 'LP_DIALOG_fileupload_progress');
            if (this.storageId == null) {
                uploadHandler.send(getUploadItemURL(), this.onCompleted.bind(this), { unzip: true});
            }
            else {
                const url = createStorageFileURL(this.storageId, this.storagePath);
                uploadHandler.send(url, this.onCompleted.bind(this));
            }
        }
        else {
            showMessageBox("파일을 먼저 선택해 주세요", "파일 업로드");
        }
    }

    onCompleted(e, response) {
        if (this.resolve) {
            this.resolve(response);
        }
        this.hide();
    }
};

function showFileUploadDialogBox(storageId, storagePath, message=null, title=null) {
    const dialogBox = new FileUploadDialogBox(storageId, storagePath);
    dialogBox.setText(message ? message : "업로드할 파일을 선택하세요", title ? title : "파일 업로드");
    return dialogBox.exec();
}