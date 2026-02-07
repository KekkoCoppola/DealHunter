# -*- coding: utf-8 -*-
"""
================================================================================
analytics.py - Modulo per Visualizzazione e Analisi Esplorativa
================================================================================

ORDINE CRONOLOGICO DEI GRAFICI (per presentazione):

FASE 1 - DATA UNDERSTANDING (dataset RAW, prima di qualsiasi modifica):
    1.1 plot_price_distribution()      → Distribuzione prezzo ORIGINALE (mostra sbilanciamento)
    1.2 plot_outlier_analysis()        → Scatter peso/schermo (identifica tablet PRIMA di rimuoverli)
    1.3 plot_feature_boxplots()        → Boxplot delle feature numeriche (distribuzione, outlier)
    1.4 plot_missing_values()          → Heatmap dei valori mancanti (zeri fittizi)

FASE 2 - POST-PREPROCESSING (dopo pulizia e feature engineering):
    2.1 plot_class_distribution()      → Distribuzione classi DOPO qcut (dimostra bilanciamento)
    2.2 plot_correlation_heatmap()     → Correlazione feature (identifica ridondanze)

FASE 3 - POST-TRAINING (valutazione modello):
    3.1 plot_model_comparison()        → Confronto accuracy tra modelli
    3.2 plot_confusion_matrix()        → Dove il modello sbaglia
    3.3 plot_feature_importance()      → Quali feature contano di più

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple


# ==============================================================================
# FASE 1: DATA UNDERSTANDING (Dataset RAW)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1.1 DISTRIBUZIONE PREZZO ORIGINALE
# ------------------------------------------------------------------------------

def plot_price_distribution(df: pd.DataFrame,
                            price_column: str = 'normalized_used_price',
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
    """
    Visualizza la distribuzione del prezzo PRIMA di applicare qcut.
    
    SCOPO: Dimostrare che i prezzi NON sono uniformi, quindi serve qcut
    (divisione per quantili) invece di cut (divisione per intervalli uguali).
    
    QUANDO USARE: Sul dataset RAW, prima di qualsiasi trasformazione.
    
    Args:
        df: DataFrame con la colonna prezzo
        price_column: Nome della colonna prezzo (default: normalized_used_price)
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Istogramma ---
    axes[0].hist(df[price_column], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Prezzo Normalizzato', fontsize=10)
    axes[0].set_ylabel('Frequenza', fontsize=10)
    axes[0].axvline(df[price_column].median(), color='red', linestyle='--', 
                    label=f'Mediana: {df[price_column].median():.2f}')
    axes[0].legend()
    
    # --- Boxplot ---
    axes[1].boxplot(df[price_column].dropna(), vert=True)
    axes[1].set_ylabel('Prezzo Normalizzato', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Grafico distribuzione prezzo salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------------------------------------
# 1.2 ANALISI OUTLIER (Scatter Peso vs Schermo)
# ------------------------------------------------------------------------------

def plot_outlier_analysis(df: pd.DataFrame,
                          weight_threshold: float = 350,
                          screen_threshold: float = 20,
                          save_path: Optional[str] = None,
                          show: bool = True) -> None:
    """
    Visualizza scatter plot peso vs schermo per identificare tablet (outlier).
    
    SCOPO: Giustificare la rimozione dei dispositivi oltre le soglie.
    I tablet (peso > 350g, schermo > 20cm) devono essere esclusi.
    
    QUANDO USARE: Sul dataset RAW, PRIMA della rimozione outlier.
    
    Args:
        df: DataFrame con colonne 'weight' e 'screen_size'
        weight_threshold: Soglia peso in grammi (default: 350g)
        screen_threshold: Soglia schermo in cm (default: 20cm)
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    sns.scatterplot(
        data=df, 
        x='weight', 
        y='screen_size', 
        hue='device_brand' if 'device_brand' in df.columns else None,
        alpha=0.6, 
        legend=False
    )
    
    # Linee di soglia
    plt.axvline(x=weight_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Soglia Peso ({weight_threshold}g)')
    plt.axhline(y=screen_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Soglia Schermo ({screen_threshold}cm)')
    
    # Evidenzia zona outlier
    plt.fill_between([weight_threshold, df['weight'].max() + 50], 
                     0, df['screen_size'].max() + 5, 
                     alpha=0.1, color='red', label='Zona Tablet (da rimuovere)')
    plt.xlabel('Peso (grammi)', fontsize=12)
    plt.ylabel('Dimensione Schermo (cm)', fontsize=12)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Grafico outlier salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------------------------------------
# 1.3 BOXPLOT FEATURE NUMERICHE
# ------------------------------------------------------------------------------

def plot_feature_boxplots(df: pd.DataFrame,
                          features: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          show: bool = True) -> None:
    """
    Visualizza boxplot delle feature numeriche per analizzare distribuzioni.
    
    SCOPO: Mostrare la distribuzione di ogni feature, identificare outlier
    e capire la scala dei valori.
    QUANDO USARE: Sul dataset RAW.
    
    Args:
        df: DataFrame con le feature
        features: Lista feature da plottare (default: feature principali)
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    """
    if features is None:
        features = ['ram', 'internal_memory', 'battery', 'weight', 
                    'screen_size', 'rear_camera_mp', 'front_camera_mp']
    
    # Filtra solo colonne presenti
    features = [f for f in features if f in df.columns]
    
    n_features = len(features)
    fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        axes[i].boxplot(df[feat].dropna())
        axes[i].set_ylabel('Valore')
    
    # Nascondi assi vuoti
    for j in range(len(features), len(axes)):
        axes[j].axis('off')
            
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Boxplot feature salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------------------------------------
# 1.4 ANALISI VALORI MANCANTI (Zeri Fittizi)
# ------------------------------------------------------------------------------

def plot_missing_values(df: pd.DataFrame,
                        columns: Optional[List[str]] = None,
                        save_path: Optional[str] = None,
                        show: bool = True) -> None:
    """
    Visualizza la percentuale di valori mancanti/zeri fittizi per colonna.
    
    SCOPO: Mostrare quali colonne hanno zeri che rappresentano dati mancanti.
    Giustifica la strategia di imputazione.
    
    QUANDO USARE: Sul dataset RAW, PRIMA dell'imputazione.
    
    Args:
        df: DataFrame originale
        columns: Colonne da analizzare (default: colonne con zeri fittizi)
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    """
    if columns is None:
        columns = ['rear_camera_mp', 'front_camera_mp', 'internal_memory', 
                   'ram', 'battery', 'weight', 'screen_size']
    
    columns = [c for c in columns if c in df.columns]
    
    # Calcola % zeri per ogni colonna
    zero_percentages = {}
    for col in columns:
        zero_count = (df[col] == 0).sum()
        zero_percentages[col] = (zero_count / len(df)) * 100
    
    # Ordina per percentuale
    sorted_cols = sorted(zero_percentages.items(), key=lambda x: x[1], reverse=True)
    cols = [x[0] for x in sorted_cols]
    percentages = [x[1] for x in sorted_cols]
    
    # Grafico
    plt.figure(figsize=(10, 5))
    colors = ['red' if p > 5 else 'orange' if p > 1 else 'green' for p in percentages]
    bars = plt.barh(cols, percentages, color=colors, edgecolor='black')
    
    # Aggiungi etichette
    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{pct:.1f}%', va='center', fontsize=9)
    
    plt.xlabel('% Valori = 0 (Zeri Fittizi)', fontsize=12)
    plt.xlim(0, max(percentages) + 5 if percentages else 10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Grafico missing values salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# FASE 2: POST-PREPROCESSING (Dopo pulizia e feature engineering)
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 DISTRIBUZIONE CLASSI (DOPO QCUT)
# ------------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame,
                            target_column: str = 'price_category',
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
    """
    Visualizza la distribuzione delle classi target DOPO l'applicazione di qcut.
    
    SCOPO: Dimostrare che le classi sono BILANCIATE grazie a qcut.
    Ogni classe dovrebbe avere circa lo stesso numero di elementi (~25%).
    
    QUANDO USARE: DOPO feature engineering (dopo create_price_categories).
    
    Args:
        df: DataFrame con la colonna target
        target_column: Nome della colonna target (default: 'price_category')
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    """
    plt.figure(figsize=(8, 5))
    
    order = ['Budget', 'Mid-Range', 'High-End', 'Premium']
    palette = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    
    ax = sns.countplot(data=df, x=target_column, palette=palette, order=order)
    
    # Aggiungi etichette con percentuali
    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        pct = count / total * 100
        ax.annotate(f'{count}\n({pct:.1f}%)', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    plt.xlabel('Fascia di Prezzo', fontsize=12)
    plt.ylabel('Numero di Dispositivi', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Grafico classi salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------------------------------------
# 2.2 HEATMAP CORRELAZIONE
# ------------------------------------------------------------------------------

def plot_correlation_heatmap(X: pd.DataFrame,
                             numeric_features: Optional[List[str]] = None,
                             threshold: float = 0.5,
                             save_path: Optional[str] = None,
                             show: bool = True) -> pd.DataFrame:
    """
    Visualizza heatmap delle correlazioni tra feature numeriche.
    
    SCOPO: Identificare feature RIDONDANTI (correlazione > 0.5).
    Es: RAM e Internal Memory sono correlate perché telefoni con
    più storage tendono ad avere più RAM.
    
    QUANDO USARE: DOPO feature engineering, PRIMA del training.
    
    Args:
        X: DataFrame delle features
        numeric_features: Lista di feature numeriche da analizzare
        threshold: Soglia per evidenziare correlazioni alte
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
        
    Returns:
        Matrice di correlazione come DataFrame
    """
    if numeric_features is None:
        numeric_features = [
            'screen_size', 'rear_camera_mp', 'front_camera_mp',
            'internal_memory', 'ram', 'battery', 'weight', 'days_used', 'model_age'
        ]
    
    available_features = [f for f in numeric_features if f in X.columns]
    corr_matrix = X[available_features].corr()
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        corr_matrix, 
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        vmin=-1, vmax=1,
        cbar_kws={'label': 'Correlazione'}
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Heatmap salvata in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return corr_matrix


def get_high_correlations(corr_matrix: pd.DataFrame, 
                          threshold: float = 0.5) -> List[Tuple[str, str, float]]:
    """
    Estrae le coppie di feature con correlazione superiore alla soglia.
    """
    high_corr = []
    features = corr_matrix.columns.tolist()
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr.append((features[i], features[j], corr_val))
    
    return sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)


def print_correlation_analysis(corr_matrix: pd.DataFrame, threshold: float = 0.5) -> None:
    """Stampa report testuale delle correlazioni significative."""
    print("\n" + "=" * 60)
    print("      ANALISI DELLE CORRELAZIONI TRA FEATURE")
    print("=" * 60)
    
    high_corr = get_high_correlations(corr_matrix, threshold)
    
    if high_corr:
        print(f"\n📊 Correlazioni superiori a {threshold}:\n")
        for feat1, feat2, corr in high_corr:
            symbol = "🔴" if abs(corr) > 0.7 else "🟡"
            print(f"   {symbol} {feat1} ↔ {feat2}: {corr:.2f}")
        
        print("\n📝 Interpretazione:")
        print("   - Feature correlate potrebbero essere ridondanti")
        print("   - Il modello gestisce automaticamente questa ridondanza")
    else:
        print(f"\n✅ Nessuna correlazione superiore a {threshold}")
    
    print("=" * 60)


# ==============================================================================
# FASE 3: POST-TRAINING (Valutazione Modello)
# ==============================================================================

# ------------------------------------------------------------------------------
# 3.1 CONFRONTO MODELLI
# ------------------------------------------------------------------------------

def plot_model_comparison(model_scores: dict,
                          save_path: Optional[str] = None,
                          show: bool = True) -> None:
    """
    Visualizza confronto tra le accuracy di diversi modelli.
    
    SCOPO: Mostrare quale modello performa meglio e giustificare la scelta.
    
    QUANDO USARE: DOPO il training di tutti i modelli.
    
    Args:
        model_scores: Dizionario {nome_modello: accuracy}
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    """
    models = list(model_scores.keys())
    scores = list(model_scores.values())
    
    best_idx = scores.index(max(scores))
    colors = ['#4CAF50' if i == best_idx else '#2196F3' for i in range(len(scores))]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, scores, color=colors, edgecolor='black')
    
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylim(0, 1)
    plt.axhline(y=0.25, color='red', linestyle='--', linewidth=1.5, 
                label='Random Guess (25%)')
    plt.xlabel('Modello', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confronto modelli salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------------------------------------
# 3.2 CONFUSION MATRIX
# ------------------------------------------------------------------------------

def plot_confusion_matrix(y_true: pd.Series,
                          y_pred: np.ndarray,
                          labels: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          show: bool = True) -> np.ndarray:
    """
    Visualizza la Confusion Matrix del modello.
    
    SCOPO: Mostrare DOVE il modello sbaglia.
    - Diagonale: predizioni corrette
    - Fuori diagonale: errori (es. Mid-Range predetto come High-End)
    
    QUANDO USARE: DOPO la predizione sul test set.
    
    Args:
        y_true: Etichette vere (ground truth)
        y_pred: Etichette predette dal modello
        labels: Ordine delle classi
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
        
    Returns:
        Matrice di confusione come numpy array
    """
    from sklearn.metrics import confusion_matrix
    
    if labels is None:
        labels = ['Budget', 'Mid-Range', 'High-End', 'Premium']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.xlabel('Predizione del Modello', fontsize=12)
    plt.ylabel('Valore Reale', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion Matrix salvata in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return cm


# ------------------------------------------------------------------------------
# 3.3 FEATURE IMPORTANCE
# ------------------------------------------------------------------------------

def plot_feature_importance(importance_df: pd.DataFrame,
                            top_n: int = 10,
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
    """
    Visualizza le feature più importanti per il modello.
    
    SCOPO: Spiegare COSA guarda il modello per decidere il prezzo (XAI).
    
    QUANDO USARE: DOPO il training del modello finale.
    
    Args:
        importance_df: DataFrame con colonne 'Feature' e 'Importance'
        top_n: Numero di feature da mostrare
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.magma(np.linspace(0.2, 0.8, len(top_features)))
    
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=top_features, 
        palette=colors
    )
    plt.xlabel('Importanza', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Feature Importance salvata in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------------------------------------
# 3.4 CONFUSION MATRIX MULTIPLA (una per ogni modello)
# ------------------------------------------------------------------------------

def plot_all_confusion_matrices(models_predictions: dict,
                                 y_true: pd.Series,
                                 labels: Optional[List[str]] = None,
                                 save_path: Optional[str] = None,
                                 show: bool = True) -> None:
    """
    Crea una griglia di Confusion Matrix, una per ogni modello.
    
    SCOPO: Confrontare visivamente dove ogni modello sbaglia.
    
    Args:
        models_predictions: Dizionario {nome_modello: y_pred}
        y_true: Etichette vere
        labels: Ordine delle classi
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    """
    from sklearn.metrics import confusion_matrix
    
    if labels is None:
        labels = ['Budget', 'Mid-Range', 'High-End', 'Premium']
    
    n_models = len(models_predictions)
    cols = 2
    rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, y_pred) in enumerate(models_predictions.items()):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[idx]
        )
        axes[idx].set_xlabel('Predizione')
        axes[idx].set_ylabel('Valore Reale')
    
    # Nascondi assi vuoti
    for j in range(len(models_predictions), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion Matrix multipla salvata in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------------------------------------
# 3.5 CONFRONTO METRICHE (Accuracy, Precision, Recall, F1)
# ------------------------------------------------------------------------------

def plot_metrics_comparison(models_metrics: dict,
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
    """
    Confronta le metriche di valutazione per tutti i modelli.
    
    SCOPO: Mostrare non solo l'accuracy, ma anche precision, recall e F1
    per ogni modello. Permette un confronto più completo.
    
    Args:
        models_metrics: Dizionario {nome_modello: {'accuracy': x, 'precision': x, 'recall': x, 'f1': x}}
        save_path: Se specificato, salva il grafico
        show: Se True, mostra il grafico
    
    Esempio:
        models_metrics = {
            'Logistic Regression': {'accuracy': 0.65, 'precision': 0.64, 'recall': 0.65, 'f1': 0.64},
            'Random Forest': {'accuracy': 0.62, 'precision': 0.61, 'recall': 0.62, 'f1': 0.61}
        }
    """
    import pandas as pd
    
    # Prepara dati per il grafico
    model_names = list(models_metrics.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Crea DataFrame per seaborn
    data = []
    for model_name, metric_values in models_metrics.items():
        for metric in metrics:
            data.append({
                'Modello': model_name,
                'Metrica': metric.capitalize(),
                'Valore': metric_values.get(metric, 0)
            })
    
    df_plot = pd.DataFrame(data)
    
    # Grafico a barre raggruppate
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    
    for i, metric in enumerate(metrics):
        values = [models_metrics[m].get(metric, 0) for m in model_names]
        bars = plt.bar(x + i * width, values, width, label=metric.capitalize(), color=colors[i])
        
        # Aggiungi etichette sopra le barre
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Modello', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x + width * 1.5, model_names, rotation=15, ha='right')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.axhline(y=0.25, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (25%)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confronto metriche salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# RIEPILOGO FUNZIONI PER NOTEBOOK
# ==============================================================================

"""
📋 ORDINE DA USARE NEL NOTEBOOK:

FASE 1 - DATA UNDERSTANDING (dataset raw):
    from analytics import plot_price_distribution, plot_outlier_analysis, plot_feature_boxplots, plot_missing_values
    
    df_raw = pd.read_csv('data/used_device_data.csv')
    
    plot_price_distribution(df_raw)       # Mostra sbilanciamento prezzi
    plot_missing_values(df_raw)           # Mostra zeri fittizi
    plot_feature_boxplots(df_raw)         # Distribuzione feature
    plot_outlier_analysis(df_raw)         # Identifica tablet
    
FASE 2 - POST-PREPROCESSING:
    from preprocessing import clean_dataset
    from feature_engineering import engineer_features
    
    df_clean = clean_dataset('data/used_device_data.csv')
    df_eng = engineer_features(df_clean)
    
    plot_class_distribution(df_eng)       # Classi bilanciate
    plot_correlation_heatmap(X)            # Correlazioni
    
FASE 3 - POST-TRAINING:
    from training import train_logistic_regression, evaluate_model
    
    plot_model_comparison({'LR': 0.65, 'RF': 0.62, 'GB': 0.63})
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(importance_df)
"""
