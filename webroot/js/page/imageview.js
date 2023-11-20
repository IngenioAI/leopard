const queryParam = getQueryParam();
const paginationNavCount = 9;

let pagination;
let currentStorageId;
let currentStoragePath;
let currentPage = 1;
let pageCount = 24;
let thumbnail_size = 160;

let rootPath = '';

function clickImage(filename) {
    showFileView(`위치: ${currentStoragePath}`, `파일보기 - ${filename}`, currentStorageId, joinPath(currentStoragePath, filename));
}

function clickFolder(dirname) {
    newPath = changeStorageDir(currentStoragePath, dirname);
    window.history.replaceState(null, "Leopard", `/ui/imageview.html?storage_id=${currentStorageId}&storage_path=${newPath}&type=Image`);
    browseDirectory(currentStorageId, newPath);
}

async function goPage() {
    const newPage = await showInputDialogBox("이동할 페이지를 입력합니다.", "페이지 이동");
    const page = parseInt(newPage);
    if (pagination && page > 0 && page <= pagination.totalPage) {
        browseDirectory(currentStorageId, currentStoragePath, page);
    }
}

function createFileItem(file) {
    let displayName = file.name;
    if (displayName.length > 14) {
        displayName = displayName.substr(0, 12) + "...";
    }
    if (file.is_dir) {
        return createElem({ name: "div", attributes: { class: "border rounded m-2", style: `display:inline-block; width:${thumbnail_size}px; height:${thumbnail_size}px;`}, children: [
            { name: "div", attributes: { class: "text-center text-nowrap", style: "cursor:pointer;" } , children: [
                { name: "span", attributes: { class: "bi bi-folder fs-4 px-2"} },
                { name: "span", text: displayName, attributes: { class: "px-4", title: file.name} }
            ],
            events: {
                click: () => clickFolder(file.name)
            }}
        ]});
    }
    else if (isImageFile(file.name)) {
        return createElem({name: "div", attributes: { class: "m-2", style: `display:inline-block; width:${thumbnail_size}px; height:${thumbnail_size}px;`}, children: [
            { name: "img", attributes: {class: "img-thumbnail rounded", src: createStorageFileURL(currentStorageId, joinPath(currentStoragePath, file.name)),
                style: "width:100%; height:100%; cursor:pointer; object-fit: contain;"},  // object-fit: cover, contain
                events: {
                    click: () => clickImage(file.name)
                }}
        ]});
    }
    else {
        return createElem({ name: "div", attributes: { class: "border rounded m-2", style: `display:inline-block; width:${thumbnail_size}px; height:${thumbnail_size}px;`}, children: [
            { name: "div", attributes: { class: "text-nowrap", style: "cursor:default" } , children: [
                { name: "span", attributes: { class: "bi bi-file-earmark fs-4 px-2"} },
                { name: "span", text: displayName, attributes: { class: "px-4", title: file.name} }
            ]}
        ]});
    }
}

async function browseDirectory(storageId, storagePath, page=1) {
    currentStorageId = storageId;
    currentStoragePath = storagePath;
    currentPage = page;

    const paths = splitPath(currentStoragePath);
    const currentPathDiv = getE("current_path");
    clearE(currentPathDiv);
    let thisPath = "/";
    for(const path of paths) {
        addE(currentPathDiv, createT("/"));
        thisPath = joinPath(thisPath, path);
        let url = `/ui/imageview.html?storage_id=${currentStorageId}&storage_path=${thisPath}&type=Image`;
        addE(currentPathDiv, createE("a", path, { href: url}))
    }

    const fileListDiv = getE("file_list");
    fileListDiv.style = `line-height:${thumbnail_size-9}px`
    const fileList = await getFileList(storageId, storagePath, page-1, pageCount);
    clearE(fileListDiv);
    for (const file of fileList.items) {
        const fileItem = createFileItem(file)
        addE(fileListDiv, fileItem);
    }

    pagination = createPagination("list_pagination", pageCount, fileList.total_count, paginationNavCount, (clickPage) => {
        browseDirectory(currentStorageId, currentStoragePath, clickPage);
    });
    if (pagination.totalPage > 1) {
        getE("btn_page").style = "display: inline";
    }
    else {
        getE("btn_page").style = "display: none";
    }
    pagination.update(currentPage);
}

async function init() {
    rootPath = queryParam.storage_path;
    browseDirectory(queryParam.storage_id, queryParam && queryParam.storage_path ? queryParam.storage_path : "/", queryParam && queryParam.page ? parseInt(queryParam.page) : 1);
}