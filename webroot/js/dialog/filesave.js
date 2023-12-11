import { ModalDialogBox } from "./dialogbox.js";
import { getE, setV, getV, clearE, addE, createE, setT } from "/js/dom_utils.js";
import { getFileList, getStorageList } from "/js/service.js";
import { joinPath, changeStorageDir, getFileIcon } from "/js/storage_utils.js";

class FileSaveDialogBox extends ModalDialogBox {
    constructor(options) {
        super("filesave");
        this.options = options;
        this.storageId = null;
        if ('defaultStoragePath' in this.options)
            this.storagePath = this.options.defaultStoragePath;
        else
            this.storagePath = '/';
    }

    async init() {
        if ('defaultFilename' in this.options) {
            setV('LP_DIALOG_filesave_filename_input', this.options.defaultFilename);
        }
        this.addEvent('LP_DIALOG_filesave_OK', 'click', this.onOK.bind(this));

        if (this.options.showPathOnly) {
            getE("LP_DIALOG_filesave_filename_div").style.display = "none";
        }
        else {
            getE("LP_DIALOG_filesave_filename_div").style.display = "";
        }

        const storageList = await getStorageList();
        const storageSelect = getE("LP_DIALOG_filesave_storage_select");
        clearE(storageSelect);
        for (const storage of storageList) {
            addE(storageSelect, createE("option", storage.name, { value: storage.id }));
        }
        if ('defaultStorageId' in this.options) {
            this.storageId = this.options.defaultStorageId;
            storageSelect.value = this.storageId;
        }
        else {
            this.storageId = storageList[0].id;
        }
        this.addEvent("LP_DIALOG_filesave_storage_select", "change", this.onStorageChange.bind(this));
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

    onStorageChange() {
        const storageSelect = getE("LP_DIALOG_filesave_storage_select");
        this.storageId = storageSelect.value;
        this.storagePath = "/";
        this.browse();
    }

    onItemClick(e, fileInfo) {
        this.storagePath = changeStorageDir(this.storagePath, fileInfo.name);
        this.browse();
    }

    onOK() {
        if (this.resolve) {
            this.resolve(`${this.storageId}:${joinPath(this.storagePath, getV('LP_DIALOG_filesave_filename_input'))}`);
        }
        this.hide();
    }
}

export function showFileSave(options={}) {
    const dialogBox = new FileSaveDialogBox(options);
    dialogBox.setText("파일 저장 위치 선택", "파일 저장");
    setT("LP_DIALOG_filesave_OK", "저장");
    return dialogBox.exec();
}

export function showSelectPath(options={}) {
    options.showPathOnly = true;
    const dialogBox = new FileSaveDialogBox(options);
    dialogBox.setText("경로 선택", "경로 지정");
    setT("LP_DIALOG_filesave_OK", "선택");
    return dialogBox.exec();
}