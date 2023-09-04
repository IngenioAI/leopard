function createPostItem(itemSpec) {
    const o = new Object();
    for (const [key, id] of itemSpec) {
        o[key] = document.getElementById(id).value;
    }
    return o;
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