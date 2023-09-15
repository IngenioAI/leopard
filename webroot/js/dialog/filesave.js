class FileSaveDialogBox extends ModalDialogBox {
    constructor(storageId, defaultFilename, defaultStoragePath) {
        super('LP_DIALOG_filesave');
        this.storageId = storageId;
        this.defaultFilename = defaultFilename;
        this.storagePath = defaultStoragePath;
    }

    onShow(e) {
        setV('LP_DIALOG_filesave_dialogbox_input', this.defaultFilename);
        this.addEvent('LP_DIALOG_filesave_dialogbox_OK', 'click', this.onOK.bind(this));
        this.browse();
    }

    addDirItem(listDiv, fileInfo) {
        const listItem = createE("a", "", {
            href: "#",
            class: "list-group-item list-group-item-action"
        }, {
            click: (e) => this.onItemClick(e, fileInfo)
        });
        const fileIcon = createE("span", "", {
            class: getFileIcon(fileInfo)
        })
        const textLabel = createE("span", fileInfo.name, {
            class: "px-2"
        });
        addE(listItem, fileIcon);
        addE(listItem, textLabel);
        addE(listDiv, listItem);
    }

    async browse() {
        setT('current_dir', this.storagePath);
        const browseInfo = await getFileList(this.storageId, this.storagePath);
        const listDiv = getE('LP_DIALOG_filesave_list');
        clearE(listDiv);
        if (this.storagePath != '/') {
            this.addDirItem(listDiv, {
                name: "..",
                is_dir: true
            });
        }
        for (const fileInfo of browseInfo.items) {
            if (fileInfo.is_dir) {
                this.addDirItem(listDiv, fileInfo);
            }
        }
    }

    onItemClick(e, fileInfo) {
        this.storagePath = changeStorageDir(this.storagePath, fileInfo.name);
        this.browse();
    }

    onOK() {
        if (this.resolve) {
            this.resolve(joinPath(this.storagePath, getV('LP_DIALOG_filesave_dialogbox_input')));
        }
        this.hide();
    }
}
function showFileSave(storageId, defaultFilename, defaultStoragePath='/') {
    const dialogBox = new FileSaveDialogBox(storageId, defaultFilename, defaultStoragePath);
    return dialogBox.exec();    
}