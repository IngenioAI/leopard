export function post(path, params, method) {
    method = method || "post";

    const form = document.createElement("form");
    form.setAttribute("method", method);
    form.setAttribute("action", path);

    for(let key in params) {
        if (Object.prototype.hasOwnProperty.call(params, key)) {
            let hiddenField = document.createElement("input");
            hiddenField.setAttribute("type", "hidden");
            hiddenField.setAttribute("name", key);
            hiddenField.setAttribute("value", params[key]);

            form.appendChild(hiddenField);
        }
    }

    document.body.appendChild(form);
    form.submit();
}

export function isStatusOK(statusCode) {
    return (statusCode >= 200 && statusCode < 300);
}

// Promise version
export function http_send(method, url, body, content_type='application/json') {
    return new Promise((resolve, reject) => {
        const httpRequest = new XMLHttpRequest();
        httpRequest.onreadystatechange = () => {
            if (httpRequest.readyState == XMLHttpRequest.DONE) {
                if (isStatusOK(httpRequest.status)) {
                    resolve(httpRequest.responseText);
                }
                else {
                    reject(httpRequest);
                }
            }
        };
        httpRequest.open(method, url);
        httpRequest.setRequestHeader('content-type', content_type);
        httpRequest.send(body);
    });
}

export function http_send_form(method, url, obj) {
    return new Promise((resolve, reject) => {
        const httpRequest = new XMLHttpRequest();
        httpRequest.onreadystatechange = () => {
            if (httpRequest.readyState == XMLHttpRequest.DONE) {
                if (isStatusOK(httpRequest.status)) {
                    resolve(httpRequest.responseText);
                }
                else {
                    reject(httpRequest.statusText);
                }
            }
        };
        httpRequest.open(method, url);

        const formData = new FormData();
        for (const [key, value] of Object.entries(obj)) {
            console.log('FORM', key, value)
            formData.append(key, value);
        }
        console.log(formData);
        httpRequest.setRequestHeader('content-type', 'multipart/form-data');
        httpRequest.send(formData);
    });
}

export function http_get(url) {
    return http_send('GET', url, null)
}

export function http_post(url, body, content_type='application/json') {
    if (typeof body == 'object') {
        body = JSON.stringify(body);
        return http_send('POST', url, body, 'application/json');
    }
    return http_send('POST', url, body, content_type)
}

export function http_put(url, body, content_type='application/json') {
    if (body != null && typeof body == 'object') {
        body = JSON.stringify(body);
        return http_send('PUT', url, body, 'application/json');
    }
    return http_send('PUT', url, body, content_type)
}

export function http_delete(url, body) {
    if (body != null && typeof body == 'object') {
        body = JSON.stringify(body)
    }
    return http_send('DELETE', url, body)
}

export class FileUploader {
    constructor(input_name) {
        this.data_files = document.getElementById(input_name).files;
        this.response = '';
    }

    send(url, metadata=null) {
        const httpRequest = new XMLHttpRequest();
        httpRequest.open("POST", url, true);
        httpRequest.onreadystatechange = () => {
            if (httpRequest.readyState == 4 && httpRequest.status == 200) {
                this.onDone(httpRequest.responseText);
            }
        }
        httpRequest.upload.addEventListener("progress", this.onUploadProgress.bind(this), false);
        httpRequest.addEventListener("load", this.onCompleted.bind(this), false);
        httpRequest.addEventListener("error", this.onError.bind(this), false);
        httpRequest.addEventListener("abort", this.onAbort.bind(this), false);
        const form_data = new FormData();
        for (const df of this.data_files) {
            form_data.append(df.name, df);
        }
        if (metadata) {
            for (const prop in metadata) {
                form_data.append('/' + prop, metadata[prop]);
            }
        }
        httpRequest.send(form_data);
    }

    getFilenames() {
        const names = [];
        for (const df of this.data_files) {
            names.push(df.name);
        }
        return names;
    }

    onUploadProgress() {
        // uploaded bytes info is in 'event.loaded' / 'event.total'
    }

    onCompleted() {

    }

    onError() {

    }

    onAbort() {

    }

    onDone(responseText) {
        this.response = responseText;
    }
}

export class FileDownloader {
    constructor(url, fileName) {
        this.url = url;
        this.fileName = fileName;
    }

    download() {
        const download_link = document.createElement("a");
        download_link.href = this.url;
        download_link.download = this.fileName;
        download_link.click();
    }
}
