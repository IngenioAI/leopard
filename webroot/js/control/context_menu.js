import { createElem } from "/js/dom_utils.js"

const bootstrap = window.bootstrap;

class ContextMenu {
    constructor(menuItems, menuHandler, buttonSpec=null) {
        const contextItems = menuItems.filter((item) => item.condition ? item.condition(item.info) : true);
        this.element = createElem({
            name: "div",
            attributes: { class: "dropdown" },
            events: { click: this.onShow.bind(this) },
            children: [{
                name: "button",
                attributes: {
                    type: "button", class: "btn LP-menu-button align-top",
                    'data-bs-toggle': "dropdown", 'data-bs-auto-close': "true", 'aria-expanded': "false"
                },
                children: [buttonSpec ? buttonSpec : {
                    name: "span",
                    attributes: { class: "bi bi-three-dots-vertical align-top"}
                }]
            }, {
                name: "ul",
                attributes: { class: "dropdown-menu" },
                children: contextItems.map((item) => {
                    return {
                        name: "li",
                        children: [{
                            name: "a",
                            text: item.title,
                            attributes: { class: "dropdown-item", href: "#" },
                            events: { click: () => this.onClickMenuItem(item.id, item.info) }
                        }]
                    }
                })
            }]
        });
        this.menuHandler = menuHandler;
        this.dropdown = null;
    }

    onShow() {
        this.dropdown = new bootstrap.Dropdown(this.element);
    }

    onClickMenuItem(menuId, info) {
        if (this.menuHandler) {
            this.menuHandler(menuId, info);
        }
        else {
            console.warn('No Handler for Menu:', menuId, info);
        }
    }

    hide() {
        if (this.dropdown)
            this.dropdown.hide();
    }
}

export { ContextMenu }