import { getE } from "/js/dom_utils.js";
import { joinPath, createStorageFileURL } from "/js/storage_utils.js";
import { runApp } from "/js/service.js";
import { showMessageBox } from "/js/dialog/messagebox.js";
import { showFileUploadDialogBox } from "/js/dialog/fileupload.js";

let encrypt_items = null;

async function analyzeText() {
    const inputText = document.getElementById("input_text").value;
    const info = await runApp('presidio', {
        type: 'analyze',
        input: inputText,
        entities: ["PERSON", "EMAIL_ADDRESS"],
        language: "en"
    });
    document.getElementById("output_text").value = JSON.stringify(info, null, 4);
}

async function anonymizeText(type) {
    const inputText = document.getElementById("input_text").value;
    if (type == 'replace') {
        const info = await runApp('presidio', {
            type: 'anonymize',
            input: inputText,
            entities: ["PERSON", "EMAIL_ADDRESS"],
            language: "en",
            operators: {
                "PERSON": {
                    type: "replace",
                    params: {
                        new_value: "***개인정보***"
                    }
                },
                "EMAIL_ADDRESS": {
                    type: "replace",
                    params: {
                        new_value: "***이메일주소***"
                    }
                }
            }
        });
        document.getElementById("output_text").value = info.text;
    }
    else if (type == 'encrypt') {
        const info = await runApp("presidio", {
            type: 'anonymize',
            input: inputText,
            entities: ["PERSON", "EMAIL_ADDRESS"],
            language: "en",
            operators: {
                "PERSON": {
                    type: "encrypt",
                    params: {
                        key: "temp_leopard_key"
                    }
                },
                "EMAIL_ADDRESS": {
                    type: "encrypt",
                    params: {
                        key: "temp_leopard_key"
                    }
                }
            }
        });
        encrypt_items = info.items;
        document.getElementById("output_text").value = info.text;
    }
}

async function deanonymizeText() {
    if (encrypt_items == null) {
        showMessageBox("익명암호화를 먼저 수행해야 합니다", "정보");
        return;
    }
    const inputText = document.getElementById("output_text").value;
    const info = await runApp("presidio", {
        type: 'deanonymize',
        input: inputText,
        result: encrypt_items,
        language: "en",
        operators: {
            "PERSON": {
                type: "decrypt",
                params: {
                    key: "temp_leopard_key"
                }
            },
            "EMAIL_ADDRESS": {
                type: "decrypt",
                params: {
                    key: "temp_leopard_key"
                }
            }
        }
    });
    document.getElementById("restore_text").value = info.text;
}

async function redactImageUpload() {
    const appStorageId = '0';
    const presidioStoragePath = 'app/presidio/data';
    const presidioOutputPath = 'app/presidio/run';
    const res = await showFileUploadDialogBox(appStorageId, presidioStoragePath);
    if (res.success) {
        const imagePath = joinPath(presidioStoragePath, res.files[0]);
        const image = new Image();
        image.src = createStorageFileURL(appStorageId, imagePath);
        image.addEventListener('load', async () => {
            const info = await runApp("presidio", {
                type: 'image_redact',
                image_path: res.files[0]
            });
            const redactedImage = new Image();
            redactedImage.src = createStorageFileURL(appStorageId, `${presidioOutputPath}/${info.image_path}`, true);
            document.getElementById("image_view").appendChild(redactedImage);
        });
        document.getElementById("image_view").innerHTML = "";
        document.getElementById("image_view").appendChild(image);
    }
}

function init() {
    getE("analyze_button").addEventListener("click", analyzeText);
    getE("anonymize_button").addEventListener("click", ()=>anonymizeText('replace'));
    getE("anonymize_encrypt_button").addEventListener("click", ()=>anonymizeText('encrypt'));
    getE("deanonymize_button").addEventListener("click", deanonymizeText);
    getE("redact_image_button").addEventListener("click", redactImageUpload);
}

init();