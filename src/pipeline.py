# -*- coding: utf-8 -*-
"""
================================================================================
pipeline.py - Pipeline Completa di Training per DealHunter
================================================================================

FLUSSO DELLA PIPELINE (9 step):
    1. PREPROCESSING: Carica dati, gestisce missing values, rimuove outlier
    2. FEATURE ENGINEERING: Crea model_age, one-hot encoding (SENZA target)
    3. PREPARE: Separa X (features) e y_price (target numerico)
    4. SPLIT: Divide train/test (80/20) con stratificazione temporanea
    5. DISCRETIZZAZIONE TARGET: fit su train (qcut), transform su test (cut)
    6. EDA: Analisi correlazione (identifica feature ridondanti)
    7. SCALING & TRAINING: StandardScaler + confronto 3 modelli
    8. TUNING: Ottimizza RF per verificare se può battere LR
    9. VALUTAZIONE E SALVATAGGIO: Sceglie il migliore e salva

NOTA IMPORTANTE SU DATA LEAKAGE:
    La discretizzazione del target (price_category) avviene DOPO lo split.
    I bin dei quantili sono calcolati SOLO sul training set (pd.qcut),
    poi applicati al test set con pd.cut. Questo previene il data leakage.

================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_PATH, MODEL_PATH, SCALER_PATH, FEATURES_PATH, BINS_PATH, PROCESSED_DATASET_PATH, ANALYTICS_DIR, PRICE_CATEGORIES
import pandas as pd
from preprocessing import clean_dataset
from feature_engineering import (
    engineer_features,
    prepare_features,
    fit_price_categories,
    transform_price_categories
)
from training import (
    split_data,
    scale_features,
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    tune_random_forest,
    evaluate_model,
    get_feature_importance,
    save_artifacts
)
from analytics import (
    plot_price_distribution,
    plot_outlier_analysis,
    plot_feature_boxplots,
    plot_missing_values,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_all_confusion_matrices,
    plot_metrics_comparison,
    print_correlation_analysis
)


def run_full_pipeline(data_path: str = DATASET_PATH,
                       save_model: bool = True,
                       save_processed: bool = True,
                       show_plots: bool = False,
                       verbose: bool = True) -> dict:
    """
    Esegue la pipeline completa di training.
    
    Il modello finale salvato sarà quello con accuracy più alta.
    Tipicamente Logistic Regression vince perché le relazioni sono lineari.
    
    FLUSSO ANTI-LEAKAGE:
    1. Preprocessing e feature engineering (senza discretizzare il target)
    2. Split train/test (con stratificazione su bin temporanei)
    3. Calcolo bin definitivi SOLO su y_train (qcut)
    4. Applicazione bin su y_test (cut)
    """
    results = {}
    
    # ==========================================================================
    # STEP 1: PREPROCESSING
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 1: PREPROCESSING")
        print("=" * 60)
    
    df_raw = pd.read_csv(data_path)  # Dataset originale per grafici Fase 1
    df = clean_dataset(data_path, verbose=verbose)
    results['rows_after_cleaning'] = len(df)
    
    # --- GRAFICI FASE 1: Data Understanding (su dati RAW) ---
    if verbose:
        print("\n📊 Generazione grafici Fase 1 (Data Understanding)...")
    
    plot_price_distribution(df_raw, save_path=os.path.join(ANALYTICS_DIR, '1_price_distribution.png'), show=show_plots)
    plot_missing_values(df_raw, save_path=os.path.join(ANALYTICS_DIR, '2_missing_values.png'), show=show_plots)
    plot_feature_boxplots(df_raw, save_path=os.path.join(ANALYTICS_DIR, '3_feature_boxplots.png'), show=show_plots)
    plot_outlier_analysis(df_raw, save_path=os.path.join(ANALYTICS_DIR, '4_outlier_analysis.png'), show=show_plots)
    
    # ==========================================================================
    # STEP 2: FEATURE ENGINEERING (senza discretizzazione target)
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 2: FEATURE ENGINEERING")
        print("=" * 60)
    
    df_engineered = engineer_features(df, verbose=verbose)
    X, y_price = prepare_features(df_engineered)
    results['n_features'] = X.shape[1]
    
    # Salva il dataset processato (con model_age, encoding, MA target numerico)
    if save_processed:
        df_engineered.to_csv(PROCESSED_DATASET_PATH, index=False)
        if verbose:
            print(f"\n💾 Dataset processato salvato: {PROCESSED_DATASET_PATH}")
    
    # ==========================================================================
    # STEP 3: SPLIT PREVENTIVO (PRIMA della discretizzazione target)
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 3: SPLIT TRAIN/TEST (prima di discretizzare il target)")
        print("=" * 60)
    
    # Discretizzazione TEMPORANEA solo per stratificazione
    # Serve a mantenere proporzioni bilanciate nello split,
    # ma NON viene usata come target definitivo.
    y_temp_strat = pd.qcut(y_price, q=4, labels=PRICE_CATEGORIES)
    
    X_train, X_test, y_train_price, y_test_price = split_data(
        X, y_price, stratify_labels=y_temp_strat
    )
    
    if verbose:
        print(f"\n📚 Training: {len(X_train)} | Test: {len(X_test)}")
    
    # ==========================================================================
    # STEP 4: DISCRETIZZAZIONE TARGET (fit su train, transform su test)
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 4: DISCRETIZZAZIONE TARGET (anti data leakage)")
        print("=" * 60)
    
    # FIT: calcola i bin SOLO dal training set
    y_train, bins = fit_price_categories(y_train_price, verbose=verbose)
    
    # TRANSFORM: applica i bin del train al test set
    y_test = transform_price_categories(y_test_price, bins, verbose=verbose)
    
    # ==========================================================================
    # STEP 5: EDA (Analisi Esplorativa)
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 5: EDA (Analisi Correlazione)")
        print("=" * 60)
    
    # --- GRAFICI FASE 2: Post-Preprocessing ---
    if verbose:
        print("\n📊 Generazione grafici Fase 2 (Post-Preprocessing)...")
    
    # Distribuzione classi (mostriamo la distribuzione sul TRAINING set)
    train_class_df = pd.DataFrame({'price_category': y_train})
    plot_class_distribution(train_class_df, save_path=os.path.join(ANALYTICS_DIR, '5_class_distribution.png'), show=show_plots)
    
    corr_save_path = os.path.join(ANALYTICS_DIR, '6_correlation_heatmap.png')
    corr_matrix = plot_correlation_heatmap(X, save_path=corr_save_path, show=show_plots)
    if verbose:
        print_correlation_analysis(corr_matrix, threshold=0.5)
    results['correlation_matrix'] = corr_matrix
    
    # ==========================================================================
    # STEP 6: SCALING
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 6: SCALING")
        print("=" * 60)
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # ==========================================================================
    # STEP 7: TRAINING - CONFRONTO 3 MODELLI
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 7: TRAINING - CONFRONTO MODELLI")
        print("=" * 60)
    
    # Logistic Regression
    if verbose:
        print("\n🔹 Training Logistic Regression...")
    log_model = train_logistic_regression(X_train_scaled, y_train)
    log_metrics = evaluate_model(log_model, X_test_scaled, y_test)
    
    # Random Forest
    if verbose:
        print("🔹 Training Random Forest...")
    rf_model = train_random_forest(X_train_scaled, y_train)
    rf_metrics = evaluate_model(rf_model, X_test_scaled, y_test)
    
    # Gradient Boosting
    if verbose:
        print("🔹 Training Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train_scaled, y_train)
    gb_metrics = evaluate_model(gb_model, X_test_scaled, y_test)
    
    # Salva risultati
    results['logistic_regression_accuracy'] = log_metrics['accuracy']
    results['random_forest_accuracy'] = rf_metrics['accuracy']
    results['gradient_boosting_accuracy'] = gb_metrics['accuracy']
    
    if verbose:
        print("\n" + "-" * 60)
        print("        RISULTATI CONFRONTO")
        print("-" * 60)
        print(f"1. Logistic Regression:    {log_metrics['accuracy']:.2%}")
        print(f"2. Random Forest:          {rf_metrics['accuracy']:.2%}")
        print(f"3. Gradient Boosting:      {gb_metrics['accuracy']:.2%}")
        print("-" * 60)
    
    # ==========================================================================
    # STEP 8: TUNING RF (per dimostrare metodologia)
    # Anche se LR vince, proviamo a ottimizzare RF per vedere se può migliorare
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 8: TUNING RF (verifica metodologica)")
        print("=" * 60)
        print("⏳ Ottimizzazione RF con GridSearchCV...\n")
    
    best_rf, best_params = tune_random_forest(
        X_train_scaled, y_train, 
        verbose=1 if verbose else 0
    )
    tuned_rf_metrics = evaluate_model(best_rf, X_test_scaled, y_test)
    results['tuned_rf_accuracy'] = tuned_rf_metrics['accuracy']
    results['best_rf_params'] = best_params
    
    if verbose:
        print(f"\n📊 RF Ottimizzato: {tuned_rf_metrics['accuracy']:.2%}")
        print(f"   Miglioramento: {(tuned_rf_metrics['accuracy'] - rf_metrics['accuracy'])*100:+.2f}%")
    
    # ==========================================================================
    # STEP 9: SELEZIONE MODELLO FINALE E SALVATAGGIO
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 9: SELEZIONE MODELLO FINALE")
        print("=" * 60)
    
    # Confronto finale (include RF ottimizzato)
    all_models = {
        'Logistic Regression': (log_model, log_metrics['accuracy']),
        'Random Forest Base': (rf_model, rf_metrics['accuracy']),
        'Random Forest Tuned': (best_rf, tuned_rf_metrics['accuracy']),
        'Gradient Boosting': (gb_model, gb_metrics['accuracy'])
    }
    
    # Trova il migliore
    best_name = max(all_models, key=lambda k: all_models[k][1])
    best_model, best_accuracy = all_models[best_name]
    
    results['best_model_name'] = best_name
    results['best_accuracy'] = best_accuracy
    
    if verbose:
        print(f"\n🏆 MODELLO VINCITORE: {best_name}")
        print(f"   Accuracy: {best_accuracy:.2%}")
        
        if 'Logistic' in best_name:
            print("\n📝 INTERPRETAZIONE:")
            print("   Le relazioni tra feature e prezzo sono LINEARI.")
            print("   Un modello semplice è preferibile (Rasoio di Occam).")
    
    # Valutazione finale del modello vincitore
    final_metrics = evaluate_model(best_model, X_test_scaled, y_test)
    results['confusion_matrix'] = final_metrics['confusion_matrix']
    results['classification_report'] = final_metrics['classification_report']
    
    if verbose:
        print(f"\n{final_metrics['classification_report']}")
    
    # --- GRAFICI FASE 3: Post-Training ---
    if verbose:
        print("\n📊 Generazione grafici Fase 3 (Post-Training)...")
    
    # Model Comparison (solo accuracy)
    model_scores = {
        'Logistic Regression': log_metrics['accuracy'],
        'Random Forest': rf_metrics['accuracy'],
        'RF Tuned': tuned_rf_metrics['accuracy'],
        'Gradient Boosting': gb_metrics['accuracy']
    }
    plot_model_comparison(model_scores, save_path=os.path.join(ANALYTICS_DIR, '7_model_comparison.png'), show=show_plots)
    
    # Confusion Matrix per ogni modello
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    all_predictions = {
        'Logistic Regression': log_metrics['predictions'],
        'Random Forest': rf_metrics['predictions'],
        'RF Tuned': tuned_rf_metrics['predictions'],
        'Gradient Boosting': gb_metrics['predictions']
    }
    plot_all_confusion_matrices(all_predictions, y_test, save_path=os.path.join(ANALYTICS_DIR, '8_all_confusion_matrices.png'), show=show_plots)
    
    # Confronto tutte le metriche
    def calc_metrics(metrics_dict):
        y_pred = metrics_dict['predictions']
        return {
            'accuracy': metrics_dict['accuracy'],
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    
    all_metrics = {
        'Logistic Regression': calc_metrics(log_metrics),
        'Random Forest': calc_metrics(rf_metrics),
        'RF Tuned': calc_metrics(tuned_rf_metrics),
        'Gradient Boosting': calc_metrics(gb_metrics)
    }
    plot_metrics_comparison(all_metrics, save_path=os.path.join(ANALYTICS_DIR, '9_metrics_comparison.png'), show=show_plots)
    
    # Feature Importance (solo per modelli tree-based)
    if hasattr(best_model, 'feature_importances_'):
        importance = get_feature_importance(best_model, X.columns)
        results['top_features'] = importance.head(10).to_dict('records')
        if verbose:
            print("\n🔝 TOP 10 FEATURE:")
            print(importance.head(10).to_string(index=False))
        plot_feature_importance(importance, save_path=os.path.join(ANALYTICS_DIR, '10_feature_importance.png'), show=show_plots)
    else:
        # Per Logistic Regression usiamo i coefficienti
        if verbose:
            print("\n🔝 TOP 10 COEFFICIENTI (valore assoluto):")
        coef_importance = get_logistic_coefficients(best_model, X.columns)
        results['top_features'] = coef_importance.head(10).to_dict('records')
        if verbose:
            print(coef_importance.head(10).to_string(index=False))
        plot_feature_importance(coef_importance, save_path=os.path.join(ANALYTICS_DIR, '10_feature_importance.png'), show=show_plots)
    
    # Salvataggio artefatti
    if save_model:
        if verbose:
            print("\n" + "=" * 60)
            print("  SALVATAGGIO ARTEFATTI")
            print("=" * 60)
        
        save_artifacts(best_model, scaler, X.columns, bins)
        
        if verbose:
            print(f"\n✅ Modello ({best_name}): {MODEL_PATH}")
            print(f"✅ Scaler: {SCALER_PATH}")
            print(f"✅ Features: {FEATURES_PATH}")
            print(f"✅ Target bins: {BINS_PATH}")
    
    return results


def get_logistic_coefficients(model, feature_names):
    """
    Estrae i coefficienti della Logistic Regression come importanza.
    
    Per LR multiclasse, prendiamo il valore assoluto medio dei coefficienti
    su tutte le classi.
    """
    import pandas as pd
    import numpy as np
    
    # model.coef_ ha shape (n_classes, n_features)
    # Prendiamo la media dei valori assoluti su tutte le classi
    mean_abs_coef = np.mean(np.abs(model.coef_), axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_coef
    })
    
    return importance_df.sort_values(by='Importance', ascending=False)


# ==============================================================================
# ESECUZIONE STANDALONE
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("      🚀 DealHunter - Training Pipeline")
    print("=" * 60)
    
    results = run_full_pipeline(
        save_model=True,
        save_processed=True,
        show_plots=False,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("      📊 RIEPILOGO FINALE")
    print("=" * 60)
    print(f"  Righe processate:           {results['rows_after_cleaning']}")
    print(f"  Numero features:            {results['n_features']}")
    print(f"  Logistic Regression:        {results['logistic_regression_accuracy']:.2%}")
    print(f"  Random Forest Base:         {results['random_forest_accuracy']:.2%}")
    print(f"  Random Forest Tuned:        {results['tuned_rf_accuracy']:.2%}")
    print(f"  Gradient Boosting:          {results['gradient_boosting_accuracy']:.2%}")
    print("-" * 60)
    print(f"  🏆 VINCITORE: {results['best_model_name']} ({results['best_accuracy']:.2%})")
    print("=" * 60)
    print("\n✅ Pipeline completata (senza data leakage)!\n")
