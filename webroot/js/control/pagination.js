import { createElem, addE, getE, clearE } from "/js/dom_utils.js"

class Pagination {
    constructor(listId, pageCount, totalCount, showCount, onClick) {
        this.listId = listId;
        this.pageCount = pageCount;
        this.totalCount = totalCount;
        this.showCount = showCount;
        this.onClick = onClick;

        this.halfshowCount = Math.floor(showCount / 2);
        this.listElem = getE(this.listId);
        this.totalPage = Math.ceil(totalCount / pageCount);

        this.currentPage = 1;
    }

    addActivePageItem(page) {
        const item = createElem({ name: "li", attributes: { class: "page-item active", style: "cursor:default;"}, children: [
            { name: "span", text: `${page}`, attributes: { class: "page-link" } }
        ]});
        addE(this.listElem, item);
    }

    addLinkPageItem(page) {
        const item = createElem({ name: "li", attributes: { class: "page-item", style: "cursor:pointer;"}, children: [
            { name: "a", text: `${page}`, attributes: { class: "page-link" }, events: { click: () => this.onClick ? this.onClick(page) : console.log("Click:", page) } }
        ]});
        addE(this.listElem, item);
    }

    addStaticPageItem(page) {
        const item = createElem({ name: "li", attributes: { class: "page-item", style: "cursor:default;"}, children: [
            { name: "span", text: `${page}`, attributes: { class: "page-link" } }
        ]});
        addE(this.listElem, item);
    }

    addPageItem(page) {
        if (page == "...") {
            this.addStaticPageItem(page);
        }
        else if (page == this.currentPage) {
            this.addActivePageItem(page);
        }
        else {
            this.addLinkPageItem(page);
        }
    }

    update(page) {
        this.currentPage = page;
        const pageList = this.listElem;
        clearE(pageList);
        if (this.totalPage > 1) {
            if (page <= 0 || page > this.totalPage) {
                console.warn("Invalid page:", page);
                return;
            }
            this.addPageItem(1);

            if (this.totalPage <= this.showCount) {
                for (let i=2; i<=this.totalPage-1; i++) {
                    this.addPageItem(i);
                }
            }
            else if (page <= this.halfshowCount+1) {
                for (let i=2; i<=this.showCount-2; i++) {
                    this.addPageItem(i);
                }
                this.addPageItem("...");
            }
            else if (page+this.halfshowCount >= this.totalPage) {
                this.addPageItem("...");
                for (let i=this.totalPage-this.showCount+3; i<=this.totalPage-1; i++) {
                    this.addPageItem(i);
                }
            }
            else {
                this.addPageItem("...");
                for (let i=page-this.halfshowCount+2; i<=page+this.halfshowCount-2; i++) {
                    this.addPageItem(i);
                }
                this.addPageItem("...");
            }
            this.addPageItem(this.totalPage);
        }
    }
}

function createPagination(listId, pageCount, totalCount, showCount=11, onClick=null) {
    return new Pagination(listId, pageCount, totalCount, showCount, onClick);
}

export { Pagination, createPagination }