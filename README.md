# 📱 DealHunter
<div align="center">
  <img src="MainLogo.png" alt="DealHunter Logo" width="400" />
  
  <h3>Cerca la trasparenza nel mercato dell'usato</h3>
  
  <p align="center">
    <a href="#-il-team">Il Team</a> •
    <a href="#-Descrizione-del-progetto">Descrizione</a> •
    <a href="#-Obiettivi-e-Key-Features">Obiettivi</a> •
    <a href="#-Struttura-della-Repository">Struttura</a> •
    <a href="#-Risultati-Sperimentali">Risultati Sperimentali</a> •
    <a href="#-Limiti-Noti">Limiti Del Modello</a> •
    <a href="#-documentazione">Documentazione</a> •
    <a href="#-roadmap-futura">Roadmap</a>
  </p>
</div>


![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-on-going)

## 👥 Full Stack Developer
*   <a href="https://github.com/KekkoCoppola"><img src="https://github.com/KekkoCoppola.png" width="25" style="border-radius: 50%; vertical-align: middle;" /> **Francesco Coppola**</a>

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
git clone [https://github.com/KekkoCoppola/DealHunter.git](https://github.com/KekkoCoppola/DealHunter.git)
cd DealHunter_Project

```

### 2. Installa le dipendenze

È consigliato l'uso di un virtual environment.

```bash
pip install -r requirements.txt

```

### 3. Avvia l'applicazione Web

Il progetto include una dashboard interattiva realizzata con **Streamlit**.

```bash
streamlit run app.py

```

L'applicazione sarà accessibile all'indirizzo: `http://localhost:8501`

## 📂 Struttura della Repository

```text
DealHunter/
├── app.py                  # Codice dell'interfaccia Web (Streamlit)
├── requirements.txt        # Librerie necessarie
├── README.md               # Documentazione del progetto
├── data/                   # Dataset (Raw e Processed)
│   └── used_device_data.csv
├── model/                  # Artefatti serializzati
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── notebooks/              # Analisi esplorativa e Training
    └── Progetto_ML_Analisi.ipynb

```

## 📊 Risultati Sperimentali

Il modello **Random Forest** (ottimizzato) ha raggiunto un'accuratezza paragonabile alla Logistic Regression (~65% su 4 classi bilanciate), evidenziando una forte linearità nelle determinanti di prezzo per le fasce estreme (Budget/Premium), con maggiore incertezza nella distinzione tra le fasce medie.

### Top 3 Fattori di Prezzo:

1. 📸 **Front Camera MP** (Selfie Camera)
2. 📷 **Rear Camera MP**
3. 🔋 **Battery Capacity**

### ⚠️ Limiti Noti

* **Data Drift**: Il dataset utilizzato copre dispositivi fino al 2020. I dispositivi successivi vengono valutati con gli standard di mercato del 2020.
* **Condizioni Estetiche**: Il modello non considera graffi o usura fisica, ma solo le specifiche hardware.


