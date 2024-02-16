// async function loadOnnxJs() {
//     return new Promise((resolve, reject) => {
//         const script = document.createElement('script');
//         script.src = "https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js";
//         script.onload = resolve;
//         script.onerror = reject;
//         document.head.appendChild(script);
//     });
// }

// document.addEventListener('DOMContentLoaded', async () => {
//     await loadOnnxJs();
//     console.log("ONNX.js loaded");

//     const canvas = document.getElementById('draw-canvas');
//     const ctx = canvas.getContext('2d');
//     let isDrawing = false;
//     let lastX = 0;
//     let lastY = 0;
//     let modelSession;

//     // Function to start drawing
//     function startDrawing(e) {
//         isDrawing = true;
//         [lastX, lastY] = [e.offsetX, e.offsetY];
//     }

//     // Function to draw on canvas
//     function draw(e) {
//         if (!isDrawing) return;
//         ctx.beginPath();
//         ctx.moveTo(lastX, lastY);
//         ctx.lineTo(e.offsetX, e.offsetY);
//         ctx.stroke();
//         [lastX, lastY] = [e.offsetX, e.offsetY];
//     }

//     // Function to stop drawing
//     function stopDrawing() {
//         isDrawing = false;
//     }

//     // Function to load the ONNX model
//     async function loadModel() {
//         try {
//             const session = new onnx.InferenceSession();
//             await session.loadModel("model.onnx");
//             modelSession = session;
//             console.log("Model loaded");
//         } catch (error) {
//             console.error('Error loading model:', error);
//         }
//     }

//     // Function to preprocess the canvas image
//     function preprocessCanvasImage(canvas) {
//         const ctx = canvas.getContext('2d');
//         const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
//         const resizedData = new Float32Array(1 * 28 * 28);

//         // Your preprocessing logic here...

//         return resizedData;
//     }

//     // Event listeners for mouse actions
//     canvas.addEventListener('mousedown', startDrawing);
//     canvas.addEventListener('mousemove', draw);
//     canvas.addEventListener('mouseup', stopDrawing);
//     canvas.addEventListener('mouseout', stopDrawing);

//     // Function to handle classification
//     document.getElementById('classify-btn').addEventListener('click', async () => {
//         if (!modelSession) {
//             console.log("Model not loaded yet");
//             return;
//         }
//         try {
//             const preprocessedInput = preprocessCanvasImage(canvas); // Preprocess the canvas image
//             const inputTensor = new onnx.Tensor(preprocessedInput, "float32", [1, 1, 28, 28]); // Create the input tensor
//             console.log("Input tensor:", inputTensor);
//             const outputMap = await modelSession.run([inputTensor]);
//             console.log("Output map object:", outputMap);
//             const outputTensor = outputMap.values().next().value;
//             console.log("Output tensor:", outputTensor);
//             // Remaining classification logic...
//         } catch (error) {
//             console.error('Error during classification:', error);
//         }
//     });

//     // Clear canvas button event listener
//     document.getElementById('clear-btn').addEventListener('click', () => {
//         ctx.clearRect(0, 0, canvas.width, canvas.height);
//     });

//     // Initialize canvas
//     ctx.fillStyle = '#FFFFFF';
//     ctx.fillRect(0, 0, canvas.width, canvas.height);
//     ctx.lineWidth = 10;
//     ctx.lineJoin = 'round';
//     ctx.lineCap = 'round';
//     ctx.strokeStyle = '#000000';

//     // Load the ONNX model
//     await loadModel();
// });








async function loadOnnxJs() {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = "https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js";
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    await loadOnnxJs();
    console.log("ONNX.js loaded");

    const canvas = document.getElementById('draw-canvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let modelSession;

    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    function draw(e) {
        if (!isDrawing) return;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    async function loadModel() {
        try {
            const session = new onnx.InferenceSession();
            await session.loadModel("model.onnx");
            modelSession = session;
            console.log("Model loaded");
        } catch (error) {
            console.error('Error loading model:', error);
        }
    }

    function preprocessCanvasImage(canvas) {
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = 28;
        offscreenCanvas.height = 28;
        const offCtx = offscreenCanvas.getContext('2d');
    
        offCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
    
        const imageData = offCtx.getImageData(0, 0, 28, 28);
        const data = imageData.data;
    
        const resizedData = new Float32Array(28 * 28);
        for (let i = 0; i < data.length; i += 4) {
            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3 / 255;
            resizedData[i / 4] = avg;
        }
    
        return resizedData;
    }
    

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);


    document.getElementById('classify-btn').addEventListener('click', async () => {
        if (!modelSession) {
            console.log("Model not loaded yet");
            return;
        }
        try {
            const preprocessedInput = preprocessCanvasImage(canvas); 
            const inputTensor = new onnx.Tensor(preprocessedInput, "float32", [1, 1, 28, 28]);
            console.log("Input tensor:", inputTensor);
            const outputMap = await modelSession.run([inputTensor]);
            console.log("Output map object:", outputMap);
            const outputTensor = outputMap.values().next().value;
            console.log("Output tensor:", outputTensor);

            const predictions = outputTensor.data;
            const predictedClass = predictions.indexOf(Math.max(...predictions));

            const predictionDisplayElement = document.getElementById('prediction-result');
            predictionDisplayElement.textContent = `Predicted Class: ${predictedClass}`;

        } catch (error) {
            console.error('Error during classification:', error);
        }
    });


    document.getElementById('clear-btn').addEventListener('click', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    });

    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 10;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000000';

    await loadModel();
});


