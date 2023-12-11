class Canvas {
    constructor(id) {
        this.canvasId = id;
    }

    init(width, height) {
        const canvas = document.getElementById(this.canvasId);
        canvas.width = width;
        canvas.height = height;
        canvas.style = "display: block; background-color: transparent";
        canvas.margin_width = 0;
        canvas.margin_height = 0;
    }

    clear() {
        const canvas = document.getElementById(this.canvasId);
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);
    }

    drawImageFit(image) {
        const canvas = document.getElementById(this.canvasId);
        let scale = 100.0;
        if (canvas.width / image.width < scale) {
            scale = canvas.width / image.width;
        }
        if (canvas.height / image.height < scale) {
            scale = canvas.height / image.height;
        }
        const context = canvas.getContext('2d');
        context.drawImage(image, 0, 0, image.width * scale, image.height * scale);
        return scale;
    }

    drawRect(x, y, w, h, width, color) {
        const canvas = document.getElementById(this.canvasId);
        const context = canvas.getContext('2d');
        context.lineWidth = width;
        context.strokeStyle = color;
        context.beginPath();
        context.rect(x, y, w, h);
        context.stroke();
        context.closePath();
    }
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.addEventListener('load', async () => {
            resolve(image);
        });
        image.addEventListener('error', async () => {
            reject(`Image load failed: ${url}`);
        });
        image.src = url;
    });
}

export { Canvas, loadImage }