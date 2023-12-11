const CodeMirror = window.CodeMirror;

export function createCodeMirrorForPython(elem) {
    const pythonConfig = {
        mode: {
            name: "python",
            version: 3,
            singleLineStringErrors: false
        },
        theme: "default",
        lineNumbers: true,
        indentUnit: 4,
        styleActiveLine: true,
        matchBrackets: true,
        autofocus: true
    };
    var editor = CodeMirror.fromTextArea(elem, pythonConfig);
    editor.on('inputRead', codeHintForPython);
    return editor;
}

export function codeHintForPython(editor, input) {
    if (input.text[0] == ';' || input.text[0] == ' ' || input.text[0] == ':') {
        return;
    }
    editor.showHint({
        hint: CodeMirror.pythonHint,
        completeSingle: false
    });
}

export function createCodeMirrorForJavascript(elem, json=false) {
    const pythonConfig = {
        mode: {
            name: "javascript",
            json: json
        },
        theme: "default",
        lineNumbers: true,
        indentUnit: 4,
        styleActiveLine: true,
        matchBrackets: true,
        autofocus: true
    };
    var editor = CodeMirror.fromTextArea(elem, pythonConfig);
    editor.on('inputRead', codeHintForPython);
    return editor;
}

export function createCodeMirror(elem) {
    const defaultConfig = {
        mode: { name: "text" },
        theme: "default",
        lineNumbers: true,
        indentUnit: 4,
        styleActiveLine: true,
        autofocus: true
    };
    var editor = CodeMirror.fromTextArea(elem, defaultConfig);
    return editor;
}