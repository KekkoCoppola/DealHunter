# 📱 DealHunter
<div align="center">
  <img src="Mainlogo.png" alt="SINTONIA Logo" width="400" />
  
  <h3>Sistema INtegrato per il Triage e l'Osservazione della salute mentale in CampaNIA</h3>
  
  <p align="center">
    <a href="#-il-team">Il Team</a> •
    <a href="#-visione-del-progetto">Visione</a> •
    <a href="#-il-problema-vs-la-soluzione">Problema vs Soluzione</a> •
    <a href="#-algoritmi-chiave">Algoritmi</a> •
    <a href="#-architettura-di-sistema">Architettura</a> •
    <a href="#-funzionalità-per-ruolo">Funzionalità</a> •
    <a href="#-documentazione">Documentazione</a> •
    <a href="#-roadmap-futura">Roadmap</a>
  </p>
</div>


![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Descrizione del Progetto
**SmartPrice** è un'applicazione di Machine Learning progettata per portare trasparenza nel mercato degli smartphone usati.
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
git clone [https://github.com/TUO_USERNAME/SmartPrice_Project.git](https://github.com/KekkoCoppola/DealHunter.git)
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
SmartPrice_Project/
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

## ⚠️ Limiti Noti

* **Data Drift**: Il dataset utilizzato copre dispositivi fino al 2020. I dispositivi successivi vengono valutati con gli standard di mercato del 2020.
* **Condizioni Estetiche**: Il modello non considera graffi o usura fisica, ma solo le specifiche hardware.

## 👥 Autori

* **[Francesco Coppola]** - *Full Stack Developer*


