import { getE, clearE, addE, createElem, getV, setV, addEvent } from "/js/dom_utils.js";
import { getStorageFileContent } from "/js/service.js";
import { codeHighlight } from "/js/code_highlight.js";
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';

async function showFile(fileType)
{
    if (fileType == 'src') {
        const contentDiv = getE('src_view');
        clearE(contentDiv);
        const content = await getStorageFileContent('0', '/app/provenance/model_src/train.py');
        codeHighlight(contentDiv, content, 'train.py');
    }
    else if (fileType == 'srcinst') {
        const contentDiv = getE('srcinst_view');
        clearE(contentDiv);
        const content = await getStorageFileContent('0', '/app/provenance/src/train.py');
        codeHighlight(contentDiv, content, 'train.py');
    }
    else if (fileType == 'logfile') {
        const contentDiv = getE('logfile_view');
        clearE(contentDiv);
        const content = await getStorageFileContent('0', '/app/provenance/src/logfile');
        codeHighlight(contentDiv, content.substring(0, 1024*64), 'logfile');
    }
}

async function drawDiagram()
{
    const element = getE("mermaid_test");
    const graphDefinition = await getStorageFileContent('0', '/app/provenance/mmd/facenet.mmd');
        const { svg } = await mermaid.render('mySvgId', graphDefinition);
    element.innerHTML = svg.replace(/( )*max-width:( 0-9\.)*px;/i, '');

    var doPan = false;
    var eventsHandler;
    var panZoom;
    var mousepos;

    eventsHandler = {
      haltEventListeners: ['mousedown', 'mousemove', 'mouseup']

      , mouseDownHandler: function (ev) {
        if (ev.target.className == "[object SVGAnimatedString]") {
          doPan = true;
          mousepos = { x: ev.clientX, y: ev.clientY }
        };
      }

      , mouseMoveHandler: function (ev) {
        if (doPan) {
          panZoom.panBy({ x: ev.clientX - mousepos.x, y: ev.clientY - mousepos.y });
          mousepos = { x: ev.clientX, y: ev.clientY };
          window.getSelection().removeAllRanges();
        }
      }

      , mouseUpHandler: function (ev) {
        doPan = false;
      }

      , init: function (options) {
        options.svgElement.addEventListener('mousedown', this.mouseDownHandler, false);
        options.svgElement.addEventListener('mousemove', this.mouseMoveHandler, false);
        options.svgElement.addEventListener('mouseup', this.mouseUpHandler, false);
      }

      , destroy: function (options) {
        options.svgElement.removeEventListener('mousedown', this.mouseDownHandler, false);
        options.svgElement.removeEventListener('mousemove', this.mouseMoveHandler, false);
        options.svgElement.removeEventListener('mouseup', this.mouseUpHandler, false);
      }
    }
    panZoom = svgPanZoom('#mySvgId', {
      zoomEnabled: true, controlIconsEnabled: true, fit: 0, center: 1, customEventsHandler: eventsHandler
    })
}

async function init() {
    mermaid.initialize({ startOnLoad: false });

    addEvent("nav-src-tab", "shown.bs.tab", () => showFile('src'));
    addEvent("nav-srcinst-tab", "shown.bs.tab", () => showFile('srcinst'));
    addEvent("nav-logfile-tab", "shown.bs.tab", () => showFile('logfile'));
    addEvent("nav-graph-tab", "shown.bs.tab", () => drawDiagram());

    showFile('src');
}

init();