# 📱 DealHunter

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## 📖 Descrizione del Progetto
**DealHunter** è un'applicazione di Machine Learning progettata per portare trasparenza nel mercato degli smartphone usati.
L'obiettivo è contrastare l'asimmetria informativa tra venditore e acquirente, fornendo una stima oggettiva della fascia di prezzo ("Budget", "Mid-Range", "High-End", "Premium") basata esclusivamente sulle specifiche tecniche del dispositivo.

Il progetto è stato sviluppato come elaborato finale per il corso di **Machine Learning 2025/26**.

## 🎯 Obiettivi e Key Features
Il progetto soddisfa i requisiti accademici attraverso una pipeline di Data Science completa:

* **Data Cleaning Avanzato**: Gestione di missing values "nascosti" (zeri fittizi in fotocamere/batterie) tramite imputazione statistica basata su Brand e Anno di rilascio.
* **Outlier Detection**: Identificazione e rimozione di dispositivi non pertinenti (Tablet > 400g) tramite analisi bivariata Peso/Schermo.
* **Feature Engineering**: Creazione di variabili derivate come `model_age` (età del dispositivo) per gestire l'obsolescenza tecnologica.
* **Model Selection**: Confronto critico tra modelli lineari (**Logistic Regression**) e non lineari (**Random Forest**), con ottimizzazione degli iperparametri (GridSearch).
* **Explainability (XAI)**: Analisi della *Feature Importance*, che ha rivelato come il comparto fotografico (Front/Rear Camera) influenzi il prezzo più della RAM.

## 🛠️ Installazione e Utilizzo

### 1. Clona la repository
```bash
<<<<<<< HEAD
git clone https://github.com/KekkoCoppola/DealHunter.git
cd DealHunter
=======
git clone [https://github.com/KekkoCoppola/DealHunter.git](https://github.com/KekkoCoppola/DealHunter.git)
cd DealHunter_Project

>>>>>>> 0ae08f57e043bd4e912ed173c2f8af5f23fac012
```

### 2. Installa le dipendenze

È consigliato l'uso di un virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Avvia l'applicazione Web

Il progetto include una dashboard interattiva realizzata con **Streamlit**.

```bash
cd src
streamlit run app.py
```

L'applicazione sarà accessibile all'indirizzo: `http://localhost:8501`

## 📂 Struttura della Repository

```text
DealHunter/
<<<<<<< HEAD
├── README.md                    # Documentazione del progetto
├── requirements.txt             # Librerie necessarie
├── data/                        # Dataset
=======
├── app.py                  # Codice dell'interfaccia Web (Streamlit)
├── requirements.txt        # Librerie necessarie
├── README.md               # Documentazione del progetto
├── data/                   # Dataset (Raw e Processed)
>>>>>>> 0ae08f57e043bd4e912ed173c2f8af5f23fac012
│   └── used_device_data.csv
├── model/                       # Artefatti serializzati
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── notebooks/                   # Analisi esplorativa e Training
│   └── dealhunter.py
└── src/                         # Codice sorgente modulare
    ├── app.py                   # Interfaccia Web (Streamlit)
    ├── config.py                # Configurazioni centralizzate
    ├── preprocessing.py         # Pulizia e imputazione dati
    ├── feature_engineering.py   # Trasformazioni e encoding
    ├── training.py              # Training e ottimizzazione modelli
    ├── predictor.py             # Classe per predizioni
    └── pipeline.py              # Pipeline completa di training
```

## 📊 Risultati Sperimentali

Il modello **Random Forest** (ottimizzato) ha raggiunto un'accuratezza paragonabile alla Logistic Regression (~65% su 4 classi bilanciate), evidenziando una forte linearità nelle determinanti di prezzo per le fasce estreme (Budget/Premium), con maggiore incertezza nella distinzione tra le fasce medie.

### Top 3 Fattori di Prezzo:

1. 📸 **Front Camera MP** (Selfie Camera)
2. 📷 **Rear Camera MP**
3. 🔋 **Battery Capacity**

## ⚠️ Limiti Noti

* **Data Drift**: Il dataset utilizzato copre dispositivi fino al 2020. I dispositivi successivi vengono valutati con gli standard di mercato del 2020.
* **Condizioni Estetiche**: Il modello non considera graffi o usura fisica, ma solo le specifiche hardware.

<<<<<<< HEAD
## 👥 Autori

* **Francesco Coppola** - *Full Stack Developer*
=======

>>>>>>> 0ae08f57e043bd4e912ed173c2f8af5f23fac012
