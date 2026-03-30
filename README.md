# 📱 DealHunter

<div align="center">
  <img src="src/img/MainLogo.png" alt="DealHunter Logo" width="400" />
  
  <h3>Cerca la trasparenza nel mercato dell'usato</h3>
  
  <p align="center">
    <a href="https://kekkocoppola.github.io/DealHunter/">🚀 Prova l'App</a> •
    <a href="#-descrizione">Descrizione</a> •
    <a href="#-come-funziona">Come Funziona</a> •
    <a href="#-installazione">Installazione</a> •
    <a href="#-struttura">Struttura</a> •
    <a href="#-risultati">Risultati</a> •
    <a href="#-limiti">Limiti</a>
  </p>
</div>

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ML](https://img.shields.io/badge/ML-Logistic%20Regression-green)
[![GitHub Pages](https://img.shields.io/badge/Live-Demo-brightgreen?logo=github&logoColor=white)](https://kekkocoppola.github.io/DealHunter/)

---

## 👤 Autore

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/KekkoCoppola">
        <img src="https://github.com/KekkoCoppola.png" width="100px;" alt="Francesco Coppola" style="border-radius: 50%; border: 2px solid #5D5CDE;"/><br />
        <sub><b>Francesco Coppola</b></sub>
      </a>
    </td>
  </tr>
</table>

---

## 📖 Descrizione

**DealHunter** è un'applicazione di Machine Learning che stima la fascia di prezzo di uno smartphone usato basandosi sulle sue specifiche tecniche.

L'obiettivo è contrastare l'**asimmetria informativa** tra venditore e acquirente, fornendo una valutazione oggettiva in 4 fasce:
- 💰 **Budget**: < 150€
- ⚖️ **Mid-Range**: 150€ - 300€
- 🚀 **High-End**: 300€ - 600€
- 💎 **Premium**: > 600€

> Progetto sviluppato per il corso di **Machine Learning 2025/26**.

---

## 🚀 Demo Live

Prova il modello direttamente nel tuo browser senza installare nulla:

[**👉 Accedi alla Web App**](https://kekkocoppola.github.io/DealHunter/)

---

## 🔬 Come Funziona

### Pipeline di Data Science

| Step | Descrizione |
|------|-------------|
| **1. Preprocessing** | Gestione missing values (zeri fittizi), imputazione statistica per Brand/Anno |
| **2. Feature Engineering** | Creazione `model_age`, One-Hot Encoding, rimozione data leakage |
| **3. EDA** | Analisi correlazione, identificazione feature ridondanti |
| **4. Training** | Confronto 3 modelli: Logistic Regression, Random Forest, Gradient Boosting |
| **5. Valutazione** | Il modello migliore viene salvato (tipicamente LR per linearità dei dati) |

### Approccio Metodologico

- **Apprendimento Supervisionato** → Classificazione multiclasse
- **Modello Finale** → Logistic Regression (~65% accuracy)
- **Perché LR?** → Le relazioni feature-prezzo sono prevalentemente lineari (Rasoio di Occam)

---

## 🛠️ Installazione

### 1. Clona la repository
```bash
git clone https://github.com/KekkoCoppola/DealHunter.git
cd DealHunter
```

### 2. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 3. Addestra il modello (una tantum)
```bash
cd src
python pipeline.py
```

### 4. Avvia l'applicazione
```bash
streamlit run app.py
```

L'app sarà disponibile su: `http://localhost:8501`

---

## 📂 Struttura

```
DealHunter/
├── analytics/
│   ├── graphics.png                         # Grafici di analisi
│   ├── ...           
├── data/
│   ├── used_device_data.csv                 # Dataset originale
│   └── used_device_data_processed.csv       # Dataset processato
├── docs/
│   ├── DealHunter.pptx                      # Presentazione Powerpoint progetto
│   ├── DealHunterPresentazione.pdf          # Presentazione Pdf Progetto
│   ├── DealHunter__Report_di_progetto.pdf   # Report Progetto
│   └── Links.txt                            # Link utili
├── model/
│   ├── logistic_regression_model.pkl        # Modello addestrato
│   ├── scaler.pkl                           # StandardScaler
│   └── feature_names.pkl                    # Lista feature
├── src/
│   ├── config.py                            # Configurazione
│   ├── preprocessing.py                     # Pulizia dati
│   ├── feature_engineering.py               # Trasformazioni
│   ├── training.py                          # Addestramento modelli
│   ├── analytics.py                         # Grafici e analisi
│   ├── pipeline.py                          # Pipeline completa
│   ├── predictor.py                         # Classe predizioni
│   └── app.py                               # Interfaccia Streamlit
├── README.md
└── requirements.txt                         # Librerie python Progetto
```

---

## 📊 Risultati

### Confronto Modelli

| Modello | Accuracy |
|---------|----------|
| **Logistic Regression** | **~65%** ✅ |
| Random Forest | ~62% |
| Gradient Boosting | ~63% |

### Top 3 Feature Influenti

1. 📸 **Front Camera MP** (fotocamera selfie)
2. 📷 **Rear Camera MP** (fotocamera principale)
3. 🔋 **Battery Capacity**

### Interpretazione

Logistic Regression batte i modelli ensemble perché le relazioni sono **lineari**:
> Più RAM → prezzo più alto | Più batteria → prezzo più alto

Non servono modelli complessi per catturare queste relazioni.

---

## ⚠️ Limiti

| Limite | Descrizione |
|--------|-------------|
| **Data Drift** | Dataset fino al 2020. Dispositivi recenti valutati con standard 2020 |
| **Hardware Only** | Non considera condizioni estetiche (graffi, usura) |

---

## 🚀 Roadmap Futura

| Priorità | Miglioramento | Descrizione |
|----------|---------------|-------------|
| 🔴 Alta | **Pipeline MLOps** | Retraining automatico periodico su dati aggiornati |
| 🔴 Alta | **Dataset Aggiornato** | Integrazione API (es. GSMArena, Kaggle) per dati 2023-2024 |
| 🟡 Media | **Condizioni Estetiche** | Input per stato batteria e condizioni fisiche |
| 🟢 Bassa | **API REST** | Endpoint per integrazione con altre applicazioni |

---

## 📚 Tecnologie

- **Python 3.9+**
- **Pandas** / **NumPy** - Manipolazione dati
- **Scikit-learn** - Machine Learning
- **Matplotlib** / **Seaborn** - Visualizzazioni
- **Streamlit** - Web App
- **Joblib** - Serializzazione modelli
