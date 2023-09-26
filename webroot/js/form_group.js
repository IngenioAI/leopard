/*
    Form Group v3
        2023.09
*/

function createFormGroup(formGroupInfo, initValue, idPrefix="") {
    const mainDiv = createElem({ name: "div" });
    formGroupInfo.forEach((formInfo) => {
        const htmlId = idPrefix + formInfo.id;
        const paramDiv = createElem({ name: "div", attributes: { class: "row mb-2" }, children: [
            { name: "label", text: formInfo.title, attributes: { class: "col-sm-3 col-form-label", for: htmlId }}
        ]});

        const formElem = createElem({ name: "div", attributes: { class: "col"}});
        let inputControl = null;
        if (formInfo.type == 'select') {
            inputControl = createElem({ name: "select", attributes: { id: htmlId, class: "form-select" },
                children: formInfo.values.map((item) => ({ name: "option", text: item }))
            });
            if (initValue != null && formInfo.id in initValue) {
                inputControl.value = initValue[formInfo.id];
            }
            else {
                inputControl.value = formInfo.default;
            }
        }
        else if (formInfo.type == 'bool') {
            let checked = false;
            if (initValue != null && formInfo.id in initValue) {
                if (initValue[formInfo.id]) {
                    checked = true;
                }
            }
            else {
                if (formInfo.default) {
                    checked = true;
                }
            }
            inputControl = createElem({ name: "div", attributes: { class: "form-check" }, children: [
                { name: "input", attributes: { id: htmlId, type: "checkbox", class: "form-check-input position-static align-middle", checked: checked ? "checked" : null }}
            ]});
        }
        else if (formInfo.type == 'strings') {
            inputControl = createElem({ name: "textarea", attributes: { id: htmlId, cols: 5, class: "form-control"}});
            if (initValue != null && formInfo.id in initValue) {
                inputControl.value = initValue[formInfo.id];
            }
            else {
                if (formInfo.default != null) {
                    inputControl.setAttribute("value", formInfo.default);
                }
            }
        }
        else {
            inputControl = createElem({ name: "input", attributes: { type: "text", id: htmlId, class: "form-control" }});
            if (initValue != null && formInfo.id in initValue) {
                inputControl.value = initValue[formInfo.id];
            }
            else {
                if (formInfo.default != null) {
                    inputControl.setAttribute("value", formInfo.default);
                }
            }
        }
        formElem.appendChild(inputControl);
        paramDiv.appendChild(formElem);
        mainDiv.appendChild(paramDiv);
    });
    return mainDiv;
}

function getFormGroupData(formGroupInfo, data=null, idPrefix="") {
    if (data == null) {
        data = {};
    }

    formGroupInfo.forEach((formInfo) => {
        const htmlId = idPrefix + formInfo.id;
        const elem = getE(htmlId);
        if (elem == null) {
            console.warn("Element not found:", htmlId);
            return;
        }

        if (formInfo.type == 'bool') {
            data[formInfo.id] = elem.checked;
        }
        else if (formInfo.type == '[number]') {
            const stringValue = elem.value;
            if (stringValue) {
                let stringArray = [];
                if (stringValue.includes(',')) {
                    stringArray = stringValue.split(',');
                }
                else {
                    stringArray = stringValue.split(' ');
                }

                data[formInfo.id] = stringArray.map((x) => {
                    return parseInt(x, 10);
                });
            }
        }
        else if (formInfo.type == 'float') {
            data[formInfo.id] = parseFloat(elem.value);
        }
        else if (formInfo.type == 'number') {
            data[formInfo.id] = parseInt(elem.value);
        }
        else {
            if (document.getElementById(htmlId).value) {
                if (formInfo.type == 'select' && elem.value == 'None') {
                    data[formInfo.id] = null;
                }
                else {
                    data[formInfo.id] = elem.value;
                }
            }
        }
    });

    return data;
}