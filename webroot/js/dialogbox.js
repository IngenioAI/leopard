class ElementEventItem {
    constructor(elementId, name, handler) {
        this.elementId = elementId;
        this.name = name;
        this.handler = handler;
    }
};

class DialogBox {
    constructor(dialogBoxId) {
        this.dialogBoxId = dialogBoxId;
        this.modal = null;
        this.eventHandlers = [];
    }

    show() {
        const dialogElement = document.getElementById(this.dialogBoxId);
        if (dialogElement) {
            this.modal = new bootstrap.Modal(dialogElement);
            this.addEvent(this.dialogBoxId, "shown.bs.modal", this.onShow.bind(this));
            this.addEvent(this.dialogBoxId, "hide.bs.modal", this.onHide.bind(this));
            this.addEvent(this.dialogBoxId, "hidden.bs.modal", this.onHidden.bind(this))
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

    onShow(e) {
    }

    onHide(e) {
    }

    // use onHide, or you should call super.onHidden(e)
    onHidden(e) {
        this.clearAllEvent();
    }
}