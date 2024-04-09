let modelSession;

import { initializeApp } from 'https://www.gstatic.com/firebasejs/9.0.0/firebase-app.js';
import { getFirestore, addDoc, collection, query, orderBy, getDocs } from 'https://www.gstatic.com/firebasejs/9.0.0/firebase-firestore.js';
import { writeBatch } from 'https://www.gstatic.com/firebasejs/9.0.0/firebase-firestore.js';

const firebaseConfig = {
    apiKey: "AIzaSyA8PpkWj7gMm5tfR-wTVOal5jt48fJC9F4",
    authDomain: "digitpro99-2ccef.firebaseapp.com",
    projectId: "digitpro99-2ccef",
    storageBucket: "digitpro99-2ccef.appspot.com",
    messagingSenderId: "64681699452",
    appId: "1:64681699452:web:c759bf7c500966914341f9",
    measurementId: "G-X5EH1Z3K4C"
};


const app = initializeApp(firebaseConfig);
const db = getFirestore(app);


async function loadModel() {
    try {
      modelSession = await new onnx.InferenceSession();
      await modelSession.loadModel("model.onnx");
      console.log("Model loaded");
    } catch (error) {
      console.error('Error loading model:', error);
    }
  }


async function loadOnnxJs() {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = "https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js";
        script.onload = () => resolve(window.onnx);
        script.onerror = () => reject(new Error('ONNX.js failed to load'));
        document.head.appendChild(script);
    });
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
        const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        resizedData[i / 4] = avg / 255;
    }

    return resizedData;
}


async function saveCanvasImage(canvas, realLabel, predictedLabel) {
    return new Promise((resolve, reject) => {
        canvas.toBlob(async function(blob) {
            const reader = new FileReader();
            reader.onloadend = async () => {
                const base64Image = reader.result;
                try {
                    await addDoc(collection(db, "images"), {
                        image: base64Image,
                        realLabel: Number(realLabel),
                        predictedLabel: predictedLabel,
                        createdAt: new Date() 
                    });
                    console.log("Image and labels saved to Firestore");
                    resolve(); 
                } catch (error) {
                    console.error("Error saving image and labels to Firestore:", error);
                    reject(error); 
                }
            };
            reader.readAsDataURL(blob);
        });
    });
}


let isImageListVisible = false;

async function listAllImages() {
    const imagesCol = collection(db, "images");
    const imageSnapshot = await getDocs(query(imagesCol, orderBy("createdAt")));
    const imageListElement = document.getElementById('image-list');

    if (isImageListVisible) {
        imageListElement.innerHTML = ''; 
        isImageListVisible = false;
    } else {
        imageListElement.innerHTML = ''; 
        imageSnapshot.forEach(doc => {
            const imageData = doc.data();

            const imageContainer = document.createElement('div');
            imageContainer.className = 'image-container';

            const imageElement = document.createElement('img');
            imageElement.src = imageData.image;
            imageElement.className = 'image-item';

            const labelElement = document.createElement('div');
            labelElement.className = 'label-item';
            labelElement.innerHTML = `<strong>Real:</strong> ${imageData.realLabel} <strong>Predicted:</strong> ${imageData.predictedLabel}`;

            imageContainer.appendChild(imageElement);
            imageContainer.appendChild(labelElement);

            imageListElement.appendChild(imageContainer);
        });
        isImageListVisible = true;
    }
}


async function clearDatabase() {
    const imagesCol = collection(db, "images");
    const snapshot = await getDocs(imagesCol);
    const batch = writeBatch(db);

    snapshot.docs.forEach((doc) => {
        batch.delete(doc.ref);
    });

    try {
        await batch.commit();
        console.log("Database cleared successfully");
        updateUIAfterDatabaseClear();
        await updateUIWithStats(); 
    } catch (error) {
        console.error("Error clearing the database:", error);
    }
}


function updateUIAfterDatabaseClear() {
    const statsTable = document.getElementById('stats-table').getElementsByTagName('tbody')[0];
    statsTable.innerHTML = '';

    document.getElementById('prediction-result').textContent = '';
    document.getElementById('image-list').innerHTML = '';
}


function setupCanvas() {
    const canvas = document.getElementById('draw-canvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });

    canvas.addEventListener('mouseup', () => {
        isDrawing = false;
    });

    canvas.addEventListener('mouseout', () => {
        isDrawing = false;
    });

    ctx.fillStyle = '#000000'; 
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = '#FFFFFF'; 
    ctx.lineWidth = 10; 
    ctx.lineJoin = 'round'; 
    ctx.lineCap = 'round'; 
}


function clearCanvas() {
    const canvas = document.getElementById('draw-canvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}


function setupEventListeners() {
    const listImagesBtn = document.getElementById('list-images-btn');
    const clearBtn = document.getElementById('clear-btn');
    const classifyBtn = document.getElementById('classify-btn');
    const realLabelInput = document.getElementById('real-label-input');

    if (listImagesBtn) {
        listImagesBtn.addEventListener('click', listAllImages);
    } else {
        console.error("Button with ID 'list-images-btn' was not found.");
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', clearCanvas);
    } else {
        console.error("Button with ID 'clear-btn' was not found.");
    }

    const clearDbBtn = document.getElementById('clear-db-btn');
    if (clearDbBtn) {
        clearDbBtn.addEventListener('click', clearDatabase);
    } else {
        console.error("Button with ID 'clear-db-btn' was not found.");
    }

    if (classifyBtn) {
        classifyBtn.addEventListener('click', async () => {
            if (!modelSession) {
                console.log("Model not loaded yet");
                return;
            }
            try {
                const preprocessedInput = preprocessCanvasImage(document.getElementById('draw-canvas'));
                const inputTensor = new onnx.Tensor(preprocessedInput, "float32", [1, 1, 28, 28]);
                const outputMap = await modelSession.run([inputTensor]);
                const outputTensor = outputMap.values().next().value;
                const predictions = outputTensor.data;
                const predictedClass = predictions.indexOf(Math.max(...predictions));
                const predictionDisplayElement = document.getElementById('prediction-result');
                predictionDisplayElement.textContent = `Predicted Class: ${predictedClass}`;
                
                const realLabel = realLabelInput.value; 
                saveCanvasImage(document.getElementById('draw-canvas'), realLabel, predictedClass)
                    .then(() => {
                        updateUIWithStats(); 
                    })
                    .catch(error => console.error('Error updating UI stats:', error));
            } catch (error) {
                console.error('Error during classification:', error);
            }        
        });
    }
}


async function getStats() {
    const imagesCol = collection(db, "images");
    const snapshot = await getDocs(imagesCol);

    const stats = {};
    for (let i = 0; i < 10; i++) {
        stats[i] = { correct: 0, incorrect: 0 };
    }

    snapshot.forEach((doc) => {
        const data = doc.data();
        const realLabel = data.realLabel.toString(); 
        if (!stats[realLabel]) {
            stats[realLabel] = { correct: 0, incorrect: 0 };
        }
        if (data.realLabel === data.predictedLabel) {
            stats[realLabel].correct++;
        } else {
            stats[realLabel].incorrect++;
        }
    });

    return stats;
}


async function updateUIWithStats() {
    const stats = await getStats();
    console.log("Stats from Firestore:", stats);
    updateStatsTable(stats);
}


function updateStatsTable(stats) {
    const table = document.getElementById('stats-table');
    table.innerHTML = '';

    Object.keys(stats).forEach(key => {
        const row = table.insertRow(-1);
        const cellDigit = row.insertCell(0);
        const cellCorrect = row.insertCell(1);
        const cellIncorrect = row.insertCell(2);

        cellDigit.textContent = `Digit ${key}`;
        cellCorrect.textContent = `Correct: ${stats[key].correct}`;
        cellIncorrect.textContent = `Incorrect: ${stats[key].incorrect}`;
    });
}


document.getElementById('canvas-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const realLabel = document.getElementById('real-label-input').value;
    if (!modelSession) {
        console.log("Model not loaded yet");
        return;
    }
    try {
        const preprocessedInput = preprocessCanvasImage(document.getElementById('draw-canvas'));
        const inputTensor = new onnx.Tensor(preprocessedInput, "float32", [1, 1, 28, 28]);
        const outputMap = await modelSession.run([inputTensor]);
        const outputTensor = outputMap.values().next().value;

        const predictions = outputTensor.data;
        const predictedClass = predictions.indexOf(Math.max(...predictions));

        const predictionDisplayElement = document.getElementById('prediction-result');
        predictionDisplayElement.textContent = `Predicted Class: ${predictedClass}`;

    } catch (error) {
        console.error('Error during classification:', error);
    }
});


async function main() {
    try {
        await loadOnnxJs();
        console.log("ONNX.js loaded");
        setupCanvas();
        setupEventListeners();
        await loadModel();
        await updateUIWithStats();
    } catch (error) {
        console.error('Error in main initialization:', error);
    }
}


document.addEventListener('DOMContentLoaded', main);
