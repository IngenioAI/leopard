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

function splitPath(path) {
    let index = 0;
    const items = [];
    if (path.startsWith("/")) {
        index = 1;
    }
    let p = path.indexOf("/", index)
    while (p > 0) {
        items.push(path.substr(index, p-index));
        index = p+1;
        p = path.indexOf("/", index);
    }
    items.push(path.substr(index));
    return items;
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
    const units = [" bytes", "KB", "MB", "GB", "TB"]
    let value = fileSize;
    let unitIndex = 0;
    while (value > 1024) {
        value /= 1024;
        unitIndex += 1;
        if (unitIndex == units.length - 1) {
            break;
        }
    }
    if (value > 100) {
        value = value.toFixed(0);
    }
    else if (value > 10) {
        value = value.toFixed(1);
    }
    else {
        value = value.toFixed(2);
    }
    return `${value}${units[unitIndex]}`;
}

function getDateString(date) {
    return date.toLocaleString();
}

function getElapsedTimeString(date) {
    const start = new Date(date);
    const end = new Date();
    const diff = (end - start) / 1000;
    const times = [
        { name: '년', milliSeconds: 60 * 60 * 24 * 365 },
        { name: '개월', milliSeconds: 60 * 60 * 24 * 30 },
        { name: '일', milliSeconds: 60 * 60 * 24 },
        { name: '시간', milliSeconds: 60 * 60 },
        { name: '분', milliSeconds: 60 },
    ];

    for (const t of times) {
        const betweenTime = Math.floor(diff / t.milliSeconds);
        if (betweenTime > 0) {
            return `${betweenTime}${t.name} 전`;
        }
    }
    return '방금 전';
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
    let url = joinPath('/api/storage/file', storageId, storagePath);
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

function getStorageName(storageId, storageList) {
    for (const storageInfo of storageList) {
        if (storageInfo.id == storageId) {
            return storageInfo.name;
        }
    }
    return storageId;
}

function getStorageId(storageName, storageList) {
    for (const storageInfo of storageList) {
        if (storageInfo.name == storageName) {
            return storageInfo.id;
        }
    }
    return storageName;
}