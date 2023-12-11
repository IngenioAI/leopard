import { createE, getE, clearE, addE, createT } from "/js/dom_utils.js";
import { getStorageList, getFileList, deleteStorageItem, uploadFileToStorage, createStorageFolder } from "/js/service.js";
import { splitPath, joinPath, changeStorageDir } from "/js/storage_utils.js";
import { isViewableFile, isEditableFile, getFileIcon, getDateString, getFileSizeString, createStorageFileURL } from "/js/storage_utils.js";
import { FileDownloader } from "/js/http.js";

import { ContextMenu } from "/js/control/context_menu.js";
import { createListGroupItem } from "/js/control/list_group.js";
import { createPagination } from "/js/control/pagination.js";
import { showFileView } from "/js/dialog/fileview.js";
import { showAskMessageBox } from "/js/dialog/ask_messagebox.js";
import { showMessageBox } from "/js/dialog/messagebox.js";
import { showCodeEditor } from "/js/dialog/code_editor.js";
import { showInputDialogBox } from "/js/dialog/input.js"
import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";

let currentStorageId = "";
let currentStoragePath = "";
let currentPage = 1;
const pageCount = 10;
let pagination = null;
const paginationItemCount = 9;
const queryParam = window.getQueryParam();

function onChangeStorage() {
    const selectStorage = document.getElementById("select_storage");
    browseDirectory(selectStorage.value, "/");
}

function createFileItem(fileInfo) {
    const menu = createContextMenu(fileInfo);
    const item = createListGroupItem([
        {
            name: "h5", attributes: { class: "mb-1" }, children: [
                { name: "span", attributes: { class: getFileIcon(fileInfo), style: "padding-right: 0.2em" } },
                { name: "span", text: fileInfo.name }
            ]
        },
        {
            name: "div", attributes: { class: "d-flex w-100 justify-content-between" }, children: [
                { name: "small", text: (fileInfo.is_dir ? "폴더" : getFileSizeString(fileInfo.size)) },
                { name: "small", text: fileInfo.mtime ? getDateString(new Date(fileInfo.mtime * 1000)) : '' }
            ]
        }],
        () => {
            if (fileInfo.is_dir) {
                const newPath = changeStorageDir(currentStoragePath, fileInfo.name);
                window.history.replaceState(null, "Leopard", `/ui/storage.html?storage_id=${currentStorageId}&storage_path=${newPath}`);
                browseDirectory(currentStorageId, newPath);
            }
        }, menu);

    return item;
}

function createContextMenu(fileInfo) {
    if (fileInfo.name == "..") {
        return createT("");
    }

    const MENU_ID = {
        DELETE: 0,
        DOWNLOAD: 1,
        VIEW: 2,
        EDIT: 3
    };
    const contextMenu = new ContextMenu([
        { id: MENU_ID.DELETE, title: "삭제", info: fileInfo },
        { id: MENU_ID.DOWNLOAD, title: "다운로드", info: fileInfo, condition: (info) => !info.is_dir },
        { id: MENU_ID.VIEW, title: "보기", info: fileInfo, condition: (info) => isViewableFile(info) },
        { id: MENU_ID.EDIT, title: "편집", info: fileInfo, condition: (info) => isEditableFile(info) }
    ],
        async (menuId, info) => {
            const storagePath = joinPath(currentStoragePath, info.name);
            if (menuId == MENU_ID.DELETE) {
                const answer = await showAskMessageBox("정말로 삭제하시겠습니까?", "파일삭제", ["확인", "취소"]);
                if (answer.index == 0) {
                    const res = await deleteStorageItem(currentStorageId, storagePath);
                    if (res.success) {
                        browseDirectory(currentStorageId, currentStoragePath);
                    }
                    else {
                        showMessageBox(`${res.errorCode}: ${res.errorMessage}`, "삭제 오류");
                    }
                }
            }
            else if (menuId == MENU_ID.DOWNLOAD) {
                const url = createStorageFileURL(currentStorageId, storagePath);
                const downloader = new FileDownloader(url, info.name);
                downloader.download();
            }
            else if (menuId == MENU_ID.VIEW) {
                showFileView(`위치: ${storagePath}`, `파일보기 - ${info.name}`, currentStorageId, storagePath);
            }
            else if (menuId == MENU_ID.EDIT) {
                const newContent = await showCodeEditor(`위치: ${storagePath}`, `파일편집 - ${info.name}`, currentStorageId, storagePath);
                if (newContent != undefined) {
                    uploadFileToStorage(currentStorageId, storagePath, newContent);
                }
            }
        }
    );
    return contextMenu.element;
}

async function browseDirectory(storageId, storagePath, page=1) {
    currentStorageId = storageId;
    currentStoragePath = storagePath;
    currentPage = page;

    const paths = splitPath(currentStoragePath);
    const currentPathDiv = getE("current_path");
    clearE(currentPathDiv);
    let thisPath = "/";
    addE(currentPathDiv, createE("a", "/", { href: `/ui/storage.html?storage_id=${currentStorageId}&storage_path=/`}))
    for(const path of paths) {
        thisPath = joinPath(thisPath, path);
        if (path != "") {
            let url = `/ui/storage.html?storage_id=${currentStorageId}&storage_path=${thisPath}`;
            addE(currentPathDiv, createE("a", path, { href: url}))
            addE(currentPathDiv, createT("/"));
        }
    }

    const fileListDiv = getE("file_list");
    const fileList = await getFileList(currentStorageId, currentStoragePath, currentPage-1, pageCount);
    clearE(fileListDiv);
    for (const file of fileList.items) {
        const fileItem = createFileItem(file)
        addE(fileListDiv, fileItem);
    }
    pagination = createPagination("list_pagination", pageCount, fileList.total_count, paginationItemCount, (clickPage) => {
        browseDirectory(currentStorageId, currentStoragePath, clickPage);
    });
    if (pagination.totalPage > 1) {
        getE("go_page_button").style = "display: inline";
    }
    else {
        getE("go_page_button").style = "display: none";
    }
    pagination.update(currentPage);
}

async function createFolder() {
    const newName = await showInputDialogBox("생성할 폴더의 이름을 입력합니다.", "새 폴더 생성", async (newName) => {
        if (!newName) {
            showMessageBox("폴더 이름을 입력하세요", "새 폴더");
            return false;
        }
        return true;
    });

    if (newName) {
        const res = await createStorageFolder(currentStorageId, joinPath(currentStoragePath, newName));
        if (!res.success) {
            if (res.errorCode == 403) {
                showMessageBox("동일한 이름의 객체가 이미 존재합니다.", "폴더 생성");
            }
            else {
                showMessageBox(`폴더 생성에 실패하였습니다. 오류코드: ${res.errorCode}`, "폴더 생성");
            }
        }
        browseDirectory(currentStorageId, currentStoragePath);
    }
}

async function uploadFile() {
    const res = await showFileUploadDialogBox(currentStorageId, currentStoragePath);
    if (res && res.success) {
        browseDirectory(currentStorageId, currentStoragePath);
    }
}

async function goPage() {
    const newPage = await showInputDialogBox("이동할 페이지를 입력합니다.", "페이지 이동");
    const page = parseInt(newPage);
    if (pagination && page > 0 && page <= pagination.totalPage) {
        browseDirectory(currentStorageId, currentStoragePath, page);
    }
}

async function init() {
    const storageList = await getStorageList();
    const selectStorage = document.getElementById("select_storage");
    while (selectStorage.length > 0) {
        selectStorage.remove(0);
    }

    for (const storageInfo of storageList) {
        const option = document.createElement("option");
        option.text = storageInfo.name;
        option.value = storageInfo.id;
        selectStorage.add(option);
    }

    if (queryParam && queryParam.storage_id) {
        selectStorage.value = queryParam.storage_id;
    }

    browseDirectory(selectStorage.value, queryParam && queryParam.storage_path ? queryParam.storage_path : "/", queryParam && queryParam.page ? parseInt(queryParam.page) : 1);

    getE("create_folder_button").addEventListener("click", createFolder);
    getE("upload_file_button").addEventListener("click", uploadFile);
    getE("select_storage").addEventListener("change", onChangeStorage);
    getE("go_page_button").addEventListener("click", goPage);
}

init();