function joinPath(...args) {
    args = args.filter(item => item != null && item != "")
    return args.map((part, i) => {
        if (i === 0) {
            return part.trim().replace(/[\/]*$/g, '')
        } else {
            return part.trim().replace(/(^[\/]*|[\/]*$)/g, '')
        }
    }).filter(x => x.length).join('/')
}

function splitStoragePath(path) {
    const items = [];
    if (path.startsWith("storage/")) {
        path = path.substr("storage/".length)
    }
    const p = path.indexOf("/")
    items.push(path.substr(0, p));
    items.push(path.substr(p+1));
    return items;
  }

function changeStorageDir(currentPath, dirName) {
    if (dirName == "..") {
        const p = currentPath.lastIndexOf("/");
        if (p > 0) {
            currentPath = currentPath.substr(0, p);
        }
        else {
            currentPath = "/";
        }
    }
    else {
        currentPath = joinPath(currentPath, dirName);
    }
    return currentPath;
}

function getFileIcon(fileInfo) {
    if (fileInfo.is_dir) {
        return "bi-folder";
    }
    if (fileInfo.name.endsWith(".py")) {
        return "bi-filetype-py";
    }
    else if (fileInfo.name.endsWith(".txt")) {
        return "bi-filetype-txt";
    }
    else if (fileInfo.name.endsWith(".csv")) {
        return "bi-filetype-csv";
    }
    else if (fileInfo.name.endsWith(".json")) {
        return "bi-filetype-json";
    }
    else if (fileInfo.name.endsWith(".md")) {
        return "bi-filetype-md";
    }
    else if (fileInfo.name.endsWith(".jpg") || fileInfo.name.endsWith(".jpeg")) {
        return "bi-filetype-jpg";
    }
    else if (fileInfo.name.endsWith(".png")) {
        return "bi-filetype-png";
    }

    return "bi-file";
}

function isViewableFile(fileInfo) {
    return isImageFile(fileInfo.name) || isTextFile(fileInfo.name);
}

function isEditableFile(fileInfo) {
    return isTextFile(fileInfo.name);
}

function isImageFile(filepath) {
    const imageFileExt = [".jpg", ".jpeg", ".png"]
    for (const ext of imageFileExt) {
        if (filepath.endsWith(ext)) {
            return true;
        }
    }
    return false;
}

function isTextFile(filepath) {
    const textFileExt = [".py", ".txt", ".csv", ".json", ".md"]
    for (const ext of textFileExt) {
        if (filepath.endsWith(ext)) {
            return true;
        }
    }
    return false;
}

function getFileSizeString(fileSize) {
    if (fileSize > 1024 * 1024 * 1024) {
        return parseInt(fileSize / 1024 / 1024 / 1024) + " GB";
    }
    else if (fileSize > 1024 * 1024) {
        return parseInt(fileSize / 1024 / 1024) + " MB";
    }
    else if (fileSize > 1024) {
        return parseInt(fileSize / 1024) + " KB";
    }
    return fileSize + " bytes";
}

function getDateString(date) {
    return date.toLocaleString();
}

function sortFileList(fileList) {
    fileList.items.sort((a, b) => {
        if (a.is_dir && !b.is_dir) {
            return -1;
        }
        else if (!a.is_dir && b.is_dir) {
            return 1;
        }
        if (a.name < b.name) {
            return -1
        }
        else if (a.name > b.name) {
            return 1;
        }
        return 0;
    });
}

function createStorageFileURL(storageId, storagePath, reload=false) {
    let url = joinPath('/api/storage_file', storageId, storagePath);
    if (reload) {
        url += `?${new Date().getTime()}`;
    }
    return url;
}

function downloadStorageFile(storageId, storagePath, fileName = null) {
    const a = document.createElement("a");
    a.href = createStorageFileURL(storageId, storagePath);
    if (fileName) {
        a.download = fileName;
    }
    else {
        const paths = storagePath.split('/');
        a.download = paths[paths.length-1];
    }
    a.click();
}