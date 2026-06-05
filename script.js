
// --- 1. Globale Variablen ---
let model;
let wordIndex = {};  // Wort -> ID
let indexWord = {};  // ID -> Wort
const SEQUENCE_LENGTH = 3; // Muss identisch zum Python-Training sein!
let autoPredictInterval;

// UI Elemente
const statusDiv = document.getElementById('status');
const inputTextArea = document.getElementById('inputText');
const btnVorhersage = document.getElementById('btnVorhersage');
const btnWeiter = document.getElementById('btnWeiter');
const predictionsList = document.getElementById('predictionsList');
const outputArea = document.getElementById('outputArea');

// --- 2. Modell und Daten laden ---
async function loadResources() {
    try {
        // Nutze hier deine exakte GitHub Pages Basis-URL!
        // Achte auf den abschließenden Schrägstrich '/' nach dem Repo-Namen.
        const BASE_URL = 'https://konstantinkoenigshofen.github.io/DeepLearning_EA3/';

        // Wörterbuch laden
        const response = await fetch(BASE_URL + 'model/woerterbuch.json');
        wordIndex = await response.json();
                        
        // Reverse-Wörterbuch erstellen
        for (let word in wordIndex) {
            indexWord[wordIndex[word]] = word;
        }

        // Modell laden
        model = await tf.loadLayersModel(BASE_URL + 'model/model.json');
        
        statusDiv.innerText = "✅ Modell und Wörterbuch erfolgreich geladen! Bereit.";
        statusDiv.style.color = "green";
        
        // Buttons aktivieren
        btnVorhersage.disabled = false;
        btnWeiter.disabled = false;
        document.getElementById('btnAuto').disabled = false;

    } catch (error) {
        console.error("Ladefehler:", error);
        statusDiv.innerText = "❌ Fehler beim Laden. Läuft der lokale Webserver?";
        statusDiv.style.color = "red";
    }
}

// --- 3. Textverarbeitung & Vorhersage (Das Herzstück) ---
async function predictNextWords() {
    const text = inputTextArea.value.trim().toLowerCase();
    if (text.length === 0) return;

    // 1. Text in Wörter aufteilen (simpler Tokenizer)
    // Ersetzt Satzzeichen durch Leerzeichen und splittet
    let words = text.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"").split(/\s+/);
    
    // 2. Wir brauchen nur die letzten 'SEQUENCE_LENGTH' Wörter
    let recentWords = words.slice(-SEQUENCE_LENGTH);
    
    // 3. Wörter in IDs umwandeln (wie in Python)
    let sequenceIds = recentWords.map(word => wordIndex[word] || 0);
    
    // 4. Padding (Vorne mit Nullen auffüllen, falls zu kurz)
    while (sequenceIds.length < SEQUENCE_LENGTH) {
        sequenceIds.unshift(0);
    }

    // 5. In einen TensorFlow Tensor umwandeln (Format: [1, 3])
    const inputTensor = tf.tensor2d([sequenceIds], [1, SEQUENCE_LENGTH]);

    // 6. Vorhersage machen
    const predictions = model.predict(inputTensor);
    
    // 7. Die Top K Wahrscheinlichkeiten holen (z.B. Top 5)
    // dataSync() holt die Werte synchron aus der GPU in ein normales JS-Array
    const probabilities = predictions.dataSync(); 
    
    // In ein Array von Objekten umwandeln zum Sortieren
    let results = [];
    for (let i = 1; i < probabilities.length; i++) { // Start bei 1, da 0 Padding ist
        if (indexWord[i]) {
            results.push({ word: indexWord[i], prob: probabilities[i] });
        }
    }
    
    // Absteigend sortieren und die Top 5 nehmen
    results.sort((a, b) => b.prob - a.prob);
    const top5 = results.slice(0, 5);

    // Speicher freigeben (sehr wichtig in TFJS!)
    inputTensor.dispose();
    predictions.dispose();

    renderPredictions(top5);
    return top5; // Wird für den "Weiter" Button gebraucht
}

// --- 4. UI Update ---
function renderPredictions(topWords) {
    outputArea.style.display = 'block';
    predictionsList.innerHTML = ''; // Vorherige löschen

    topWords.forEach(item => {
        const percent = (item.prob * 100).toFixed(1);
        const chip = document.createElement('div');
        chip.className = 'prediction-chip';
        chip.innerHTML = `<strong>${item.word}</strong> <span style="font-size:0.8em">(${percent}%)</span>`;
        
        // I1) Klick auf ein Wort hängt es an den Text an
        chip.onclick = () => appendWord(item.word);
        predictionsList.appendChild(chip);
    });
}

function appendWord(word) {
    inputTextArea.value += " " + word;
    predictNextWords(); // Sofort neue Vorhersage triggern (laut Aufgabe I1)
}

// --- 5. Event Listener ---
btnVorhersage.onclick = predictNextWords;

btnWeiter.onclick = async () => {
    const topWords = await predictNextWords();
    if (topWords && topWords.length > 0) {
        appendWord(topWords[0].word); // Nimmt das wahrscheinlichste Wort
    }
};

document.getElementById('btnReset').onclick = () => {
    inputTextArea.value = '';
    outputArea.style.display = 'none';
};

// App starten
loadResources();

