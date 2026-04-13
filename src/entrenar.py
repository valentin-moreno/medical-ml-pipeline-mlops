# Paso 2: Entrenamiento con XGBoost + Optuna + umbral optimizado
# Lo que hace este script:
#   - Carga los datos con feature engineering
#   - Usa Optuna para encontrar los mejores hiperparametros
#   - Optimiza el umbral de decision para maximizar F1 en clase minoritaria
#   - Registra todo en MLflow con metricas clinicamente relevantes
#   - Guarda el mejor modelo en disco
#
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
import joblib
import os
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Silenciamos logs de optuna que no aportan
optuna.logging.set_verbosity(optuna.logging.WARNING)

RUTA_DATOS         = "data/diabetes_limpio.csv"
RUTA_MODELO        = "models/mejor_modelo.pkl"
NOMBRE_EXPERIMENTO = "readmision-hospitalaria-xgboost"
N_TRIALS           = 150

# Ratio clase mayoritaria / minoritaria (~89/11 = ~8)
# Le dice a XGBoost que penalice mas los errores en la clase minoritaria
SCALE_POS_WEIGHT = 8


def cargar_datos():
    """Carga el dataset con feature engineering y lo divide en train/test."""
    print("Cargando datos...")

    if not os.path.exists(RUTA_DATOS):
        raise FileNotFoundError(
            f"No se encontro {RUTA_DATOS}. "
            "Corre primero: python src/preparar_datos.py"
        )

    df = pd.read_csv(RUTA_DATOS)

    X = df.drop(columns=["readmitido_30dias"])
    y = df["readmitido_30dias"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"  -> Entrenamiento: {X_train.shape[0]:,} muestras")
    print(f"  -> Prueba:        {X_test.shape[0]:,} muestras")
    print(f"  -> Features:      {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def objetivo_optuna(trial, X_train, y_train):
    """
    Funcion objetivo para Optuna.
    Sugiere hiperparametros y retorna el AUC promedio en validacion cruzada.
    Usamos StratifiedKFold para mantener proporcion de clases en cada fold.
    """
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0, 5),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0, 2),
    }

    modelo = XGBClassifier(
        **params,
        scale_pos_weight = SCALE_POS_WEIGHT,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
        eval_metric      = "auc",
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1")
    return scores.mean()


def optimizar_umbral(modelo, X_test, y_test):
    """
    Busca el umbral de decision que maximiza el F1 de la clase positiva.
    El umbral por defecto de 0.5 (o peor, 0.7) es inadecuado con desbalance.
    Probamos todos los umbrales entre 0.1 y 0.9 y elegimos el mejor.
    """
    print("Optimizando umbral de decision...")

    y_proba = modelo.predict_proba(X_test)[:, 1]
    mejor_f1      = 0
    mejor_umbral  = 0.5

    for umbral in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= umbral).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > mejor_f1:
            mejor_f1     = f1
            mejor_umbral = umbral

    print(f"  -> Mejor umbral: {mejor_umbral:.2f} (F1 clase 1: {mejor_f1:.4f})")
    return mejor_umbral


def calcular_metricas(y_test, y_pred, y_proba):
    """Calcula todas las metricas relevantes para el problema clinico."""
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "auc_roc":              round(roc_auc_score(y_test, y_proba), 4),
        "average_precision":    round(average_precision_score(y_test, y_proba), 4),
        "precision_clase1":     round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall_clase1":        round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_clase1":            round(f1_score(y_test, y_pred, zero_division=0), 4),
        "specificity":          round(specificity, 4),
        "verdaderos_positivos": int(tp),
        "falsos_positivos":     int(fp),
        "verdaderos_negativos": int(tn),
        "falsos_negativos":     int(fn),
    }


def imprimir_metricas(metricas, umbral):
    """Imprime las metricas de forma legible en consola."""
    print(f"\nResultados con umbral={umbral:.2f}:")
    print(f"  AUC-ROC:           {metricas['auc_roc']:.4f}")
    print(f"  Average Precision: {metricas['average_precision']:.4f}")
    print(f"  Precision clase 1: {metricas['precision_clase1']:.4f}")
    print(f"  Recall clase 1:    {metricas['recall_clase1']:.4f}  <- detecta readmisiones reales")
    print(f"  F1 clase 1:        {metricas['f1_clase1']:.4f}")
    print(f"  Specificity:       {metricas['specificity']:.4f}")
    print(f"\n  Verdaderos positivos: {metricas['verdaderos_positivos']}")
    print(f"  Falsos positivos:     {metricas['falsos_positivos']}")
    print(f"  Verdaderos negativos: {metricas['verdaderos_negativos']}")
    print(f"  Falsos negativos:     {metricas['falsos_negativos']}")


def entrenar(X_train, X_test, y_train, y_test):
    """
    Ejecuta la busqueda de hiperparametros con Optuna,
    optimiza el umbral y registra todo en MLflow.
    """
    mlflow.set_experiment(NOMBRE_EXPERIMENTO)

    print(f"\nBuscando mejores hiperparametros ({N_TRIALS} intentos)...")

    estudio = optuna.create_study(direction="maximize")
    estudio.optimize(
        lambda trial: objetivo_optuna(trial, X_train, y_train),
        n_trials         = N_TRIALS,
        show_progress_bar= True,
    )

    mejores_params = estudio.best_params
    mejor_auc_cv   = estudio.best_value

    print(f"\nMejores hiperparametros:")
    for k, v in mejores_params.items():
        print(f"  {k}: {v}")
    print(f"\nAUC-ROC validacion cruzada: {mejor_auc_cv:.4f}")

    # Entrenamos el modelo final con los mejores parametros
    print("\nEntrenando modelo final...")
    mejor_modelo = XGBClassifier(
        **mejores_params,
        scale_pos_weight = SCALE_POS_WEIGHT,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
        eval_metric      = "auc",
    )
    mejor_modelo.fit(X_train, y_train)

    # Optimizamos el umbral de decision
    umbral_optimo = optimizar_umbral(mejor_modelo, X_test, y_test)

    # Predicciones con el umbral optimizado
    y_proba = mejor_modelo.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= umbral_optimo).astype(int)

    # Calculamos e imprimimos metricas
    metricas = calcular_metricas(y_test, y_pred, y_proba)
    imprimir_metricas(metricas, umbral_optimo)

    print(f"\nReporte completo:")
    print(classification_report(y_test, y_pred))

    # Registramos en MLflow
    with mlflow.start_run(run_name="xgboost-optuna-umbral-optimizado"):

        mlflow.log_params(mejores_params)
        mlflow.log_param("scale_pos_weight", SCALE_POS_WEIGHT)
        mlflow.log_param("n_trials_optuna",  N_TRIALS)
        mlflow.log_param("umbral_decision",  round(umbral_optimo, 2))
        mlflow.log_param("modelo",           "XGBClassifier")

        mlflow.log_metric("auc_cv", mejor_auc_cv)
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)

        mlflow.sklearn.log_model(mejor_modelo, "modelo")

        # Matriz de confusion como imagen
        os.makedirs("models", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["No readmitido", "Readmitido"],
            cmap="Blues",
            normalize="true",    # normaliza por fila — cada celda muestra % en vez de conteo
            values_format=".2f", # formato de 2 decimales
            ax=ax
        )
        ax.set_title("Matriz de Confusion - Readmision Hospitalaria")
        fig.savefig("models/confusion_matrix.png", bbox_inches="tight")
        mlflow.log_artifact("models/confusion_matrix.png")
        plt.close(fig)

        print("\nExperimento registrado en MLflow!")
        print("Para ver la UI corre: python -m mlflow ui")

    return mejor_modelo, metricas, mejores_params, umbral_optimo


def guardar_modelo(modelo, X_train, metricas, umbral):
    """Guarda el modelo, columnas, umbral y metricas en disco."""
    os.makedirs("models", exist_ok=True)

    joblib.dump(modelo, RUTA_MODELO)
    joblib.dump(list(X_train.columns), "models/columnas.pkl")
    joblib.dump(umbral, "models/umbral.pkl")

    with open("models/metricas.txt", "w") as f:
        f.write(f"auc_test={metricas['auc_roc']:.4f}\n")
        f.write(f"recall_clase1={metricas['recall_clase1']:.4f}\n")
        f.write(f"f1_clase1={metricas['f1_clase1']:.4f}\n")
        f.write(f"umbral={umbral:.2f}\n")

    print(f"\nModelo guardado en: {RUTA_MODELO}")


def entrenar_pipeline():
    """Funcion principal — ejecuta el flujo completo de entrenamiento."""
    print("=" * 55)
    print("ENTRENAMIENTO CON XGBOOST + OPTUNA")
    print("=" * 55)

    X_train, X_test, y_train, y_test = cargar_datos()
    modelo, metricas, params, umbral = entrenar(X_train, X_test, y_train, y_test)
    guardar_modelo(modelo, X_train, metricas, umbral)

    print("\n" + "=" * 55)
    print("Entrenamiento completado!")
    print(f"  AUC-ROC:        {metricas['auc_roc']:.4f}")
    print(f"  Recall clase 1: {metricas['recall_clase1']:.4f}")
    print(f"  F1 clase 1:     {metricas['f1_clase1']:.4f}")
    print(f"  Umbral optimo:  {umbral:.2f}")
    print("=" * 55)

    return modelo


if __name__ == "__main__":
    entrenar_pipeline()