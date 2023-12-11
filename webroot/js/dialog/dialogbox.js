const bootstrap = window.bootstrap;

class ElementEventItem {
    constructor(elementId, name, handler) {
        this.elementId = elementId;
        this.name = name;
        this.handler = handler;
    }
}

class DialogBox {
    constructor(dialogBoxId, options={}) {
        if (dialogBoxId.indexOf("LP_DIALOG_") == 0) {
            this.dialogBoxId = dialogBoxId;
        }
        else {
            this.dialogBoxId = "LP_DIALOG_" + dialogBoxId;
        }
        this.options = options;
        this.modal = null;
        this.eventHandlers = [];
    }

    async init() {

    }

    setText(message, title) {
        if (title) {
            const titleDiv = document.getElementById(this.dialogBoxId + '_title');
            if (titleDiv)
                titleDiv.innerHTML = title;
        }
        if (message) {
            const messageDiv = document.getElementById(this.dialogBoxId + '_message');
            if (messageDiv)
                messageDiv.innerHTML = message;
        }
    }

    async show() {
        const dialogElement = document.getElementById(this.dialogBoxId);
        if (dialogElement) {
            this.modal = new bootstrap.Modal(dialogElement);
            this.addEvent(this.dialogBoxId, "shown.bs.modal", this.onShow.bind(this));
            this.addEvent(this.dialogBoxId, "hide.bs.modal", this.onHide.bind(this));
            this.addEvent(this.dialogBoxId, "hidden.bs.modal", this.onHidden.bind(this));
            if (this.options && "dialog_size" in this.options) {
                dialogElement.firstElementChild.classList.add("modal-" + this.options["dialog_size"]);
            }
            await this.init();
            this.modal.show();
        }
        else {
            console.error('DialogBox id not found:', this.dialogBoxId);
        }
    }

    hide() {
        this.modal.hide();
    }

    addEvent(elementId, name, handler) {
        const element = document.getElementById(elementId);
        if (element) {
            element.addEventListener(name, handler);
            this.eventHandlers.push(new ElementEventItem(elementId, name, handler));
        }
    }

    removeEvent(elementId, name) {
        const element = document.getElementById(elementId);
        if (element) {
            for (let i = 0; i < this.eventHandlers.length; i++) {
                const eventItem = this.eventHandlers[i];
                if (eventItem.elementId == elementId && eventItem.name == name) {
                    element.removeEventListener(name, this.eventHandlers[name]);
                    this.eventHandlers.splice(i, 1);
                }
            }
        }
    }

    clearAllEvent() {
        for (const eventItem of this.eventHandlers) {
            const element = document.getElementById(eventItem.elementId);
            if (element) {
                element.removeEventListener(eventItem.name, eventItem.handler);
            }
        }
        this.eventHandlers = [];
    }

    onShow() {
    }

    onHide() {
    }

    // use onHide, or you should call super.onHidden(e)
    onHidden() {
        this.clearAllEvent();
    }
}

class ModalDialogBox extends DialogBox {
    constructor(dialogBoxId, options={}) {
        super(dialogBoxId, options);
        this.resolve = null;
        this.reject = null;
        this.resolved = true;
    }

    exec(...args) {
        return new Promise((resolve, reject) => {
            this.resolve = resolve;
            this.reject = reject;
            this.resolved = false;
            this.show(args);
        });
    }

    onHidden(e) {
        if (!this.resolved) {
            this.close();
        }
        super.onHidden(e);
    }

    close(...args) {
        if (this.resolved) {
            console.warn("ModalDialog already resolved")
        }
        this.resolve(...args);
        this.resolved = true;
        this.hide();
    }
}

export { DialogBox, ModalDialogBox }