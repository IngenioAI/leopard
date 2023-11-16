class FileViewDialogBox extends DialogBox {
    constructor(storageId, storagePath) {
        super("fileview");
        this.storageId = storageId;
        this.storagePath = storagePath;
        if (isImageFile(this.storagePath)) {
            this.options.dialog_size = "lg";
        }
        else {
            this.options.dialog_size = "xl";
        }
    }

    async init() {
        if (isImageFile(this.storagePath)) {
            const contentDiv = getE('LP_DIALOG_fileview_content');
            clearE(contentDiv);
            const url = createStorageFileURL(this.storageId, this.storagePath);
            const image = await loadImage(url);
            addE(contentDiv, createE("canvas", "", {id: "LP_DIALOG_fileview_canvas"}));
            const canvas = new Canvas("LP_DIALOG_fileview_canvas");
            canvas.init(600, Math.floor(600 * image.height / image.width));
            canvas.drawImageFit(image);
        }
        else {
            const contentDiv = getE('LP_DIALOG_fileview_content');
            clearE(contentDiv);
            const content = await getStorageFileContent(this.storageId, this.storagePath);
            codeHighlight(contentDiv, content, this.storagePath);
        }
    }
}

function showFileView(message, title, storageId, storagePath) {
    const dialogBox = new FileViewDialogBox(storageId, storagePath);
    dialogBox.setText(message, title);
    dialogBox.show();
}