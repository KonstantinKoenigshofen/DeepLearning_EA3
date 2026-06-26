
// --- 1. Globale Variablen ---
let model;
let wordIndex = {};  // Wort -> ID
let indexWord = {};  // ID -> Wort
const SEQUENCE_LENGTH = 15; // Muss identisch zum Python-Training sein!
let autoPredictInterval;
let isAutoPredicting = false; // Flag, um die Schleife zu stoppen
const MAX_AUTO_WORDS = 10;    // Laut Aufgabe: bis zu 10 Wörter

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
        
        statusDiv.innerText = "Modell und Wörterbuch geladen.";
        statusDiv.style.color = "green";
        
        // Buttons aktivieren
        btnVorhersage.disabled = false;
        btnWeiter.disabled = false;
        document.getElementById('btnAuto').disabled = false;

    } catch (error) {
        console.error("Ladefehler:", error);
        statusDiv.innerText = "Fehler beim Laden.";
        statusDiv.style.color = "red";
    }
}

// --- 3. Textverarbeitung & Vorhersage (Das Herzstück) ---
async function predictNextWords() {
    let text = inputTextArea.value.trim().toLowerCase();
    if (text.length === 0) return;

    // 1. Text in Wörter aufteilen (simpler Tokenizer)
    // ALT:
    // let words = text.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"").split(/\s+/);
    
    // NEU: Säubert den Text exakt so wie das Python-Backend!
    // Ersetzt alles, was KEIN Buchstabe (a-z, A-Z) und KEIN Umlaut ist, durch ein Leerzeichen.
    text = text.replace(/[^a-zA-ZäöüÄÖÜß\s]/g, " ");
    let words = text.split(/\s+/).filter(word => word.length > 0);
    
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
    // Hilfsfunktion: Wartet X Millisekunden (für eine schöne Animation beim Auto-Schreiben)
    const delay = ms => new Promise(res => setTimeout(res, ms));

    // I1) Vorhersage
    btnVorhersage.onclick = predictNextWords;
    
    // I2) Weiter 
    // Greedy Decoding
    
    btnWeiter.onclick = async () => {
        const topWords = await predictNextWords();
        if (topWords && topWords.length > 0) {
            appendWord(topWords[0].word); 
        }
    };
    // Zufällige Auswahl
    /*
    btnWeiter.onclick = async () => {
            const topWords = await predictNextWords();
            if (topWords && topWords.length > 0) {
                // Wählt zufällig eines der Top-3 Wörter aus, um kreativer zu sein!
                const k = Math.min(3, topWords.length);
                const randomIndex = Math.floor(Math.random() * k);
                appendWord(topWords[randomIndex].word); 
            }
        };*/

    // I3) Auto
    document.getElementById('btnAuto').onclick = async () => {
        isAutoPredicting = true;

        // UX: Andere Buttons deaktivieren, Stopp aktivieren
        document.getElementById('btnAuto').disabled = true;
        document.getElementById('btnStopp').disabled = false;
        btnVorhersage.disabled = true;
        btnWeiter.disabled = true;

        for (let i = 0; i < MAX_AUTO_WORDS; i++) {
            if (!isAutoPredicting) break; // Bricht ab, wenn der User "Stopp" klickt

            const topWords = await predictNextWords();
            
            if (topWords && topWords.length > 0) {
                // Wort direkt an den Text anhängen
                inputTextArea.value += " " + topWords[0].word; //-> noch für Greedy Decoding
                /* zufällige Auswahl
                const k = Math.min(3, topWords.length);
                const randomIndex = Math.floor(Math.random() * k);
                inputTextArea.value += " " + topWords[randomIndex].word;*/
                
                // 400 Millisekunden warten (UX: Man kann beim Schreiben zusehen)
                await delay(400); 
            } else {
                break; // Beenden, falls keine Vorhersage möglich ist (z.B. Feld leer)
            }
        }

        // Wenn die 10 Wörter erreicht sind, oder abgebrochen wurde: Aufräumen
        stopAutoPredict();
    };

    // I3) Stopp
    document.getElementById('btnStopp').onclick = () => {
        isAutoPredicting = false; // Das teilt der Schleife oben mit, dass sie stoppen soll
    };

    // Hilfsfunktion zum Zurücksetzen der Buttons nach dem Auto-Lauf
    function stopAutoPredict() {
        isAutoPredicting = false;
        
        // UX: Buttons wieder in Normalzustand
        document.getElementById('btnAuto').disabled = false;
        document.getElementById('btnStopp').disabled = true;
        btnVorhersage.disabled = false;
        btnWeiter.disabled = false;

        // Am Ende einmal die Chips für das letzte Wort anzeigen
        predictNextWords();
    }

// I4) Reset
    document.getElementById('btnReset').onclick = () => {
        // 1. Textfeld leeren
        inputTextArea.value = '';
        
        // 2. Vorhersage-Chips ausblenden
        outputArea.style.display = 'none';
        
        // 3. Falls gerade die "Auto"-Schleife rattert, abbrechen
        stopAutoPredict(); 

        // 4. NEU: Das Netzwerk offiziell zurücksetzen!
        // Leert das interne Kurzzeitgedächtnis (Hidden States) des Modells.
        if (model && model.resetStates) {
            model.resetStates();
            console.log("Netzwerk-Zustand wurde zurückgesetzt.");
        }

    }
// App starten
loadResources();

