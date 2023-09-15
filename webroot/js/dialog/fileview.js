class FileViewDialogBox extends DialogBox {
    constructor(storageId, storagePath) {
        super('LP_DIALOG_fileview');
        this.storageId = storageId;
        this.storagePath = storagePath;
    }

    async init() {
        if (isImageFile(this.storagePath)) {
            const contentDiv = getE('LP_DIALOG_fileview_content');
            clearE(contentDiv);
            const url = createStorageFileURL(this.storageId, this.storagePath)
            const image = createE("img", "", { src: url, style: "width:100%; height:100%" });
            addE(contentDiv, image);
        }
        else {
            const contentDiv = getE('LP_DIALOG_fileview_content');
            clearE(contentDiv);
            const content = await getStorageFileContent(this.storageId, this.storagePath);
            const text = createE("pre", content, { class: "border p-2 bg-light" });
            addE(contentDiv, text);
        }
    }
}
function showFileView(message, title, storageId, storagePath) {
    const dialogBox = new FileViewDialogBox(storageId, storagePath);
    dialogBox.setText(message, title);
    dialogBox.show();
}