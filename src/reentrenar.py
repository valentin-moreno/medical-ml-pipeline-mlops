# Automatizacion: Reentrenamiento automatico con XGBoost
# Lo que hace este script:
#   - Revisa si llegaron datos nuevos a la carpeta data/nuevos/
#   - Si hay datos nuevos, los une con los existentes y reentrena el modelo
#   - Registra el nuevo entrenamiento en MLflow
#   - Solo reemplaza el modelo si el nuevo es mejor
#
# Para automatizarlo:
#   cron (Linux/Mac): 0 2 * * * python src/reentrenar.py
#   Task Scheduler (Windows): ejecutar diariamente a las 2am
#
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA
 
import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
 
RUTA_DATOS_ACTUALES   = "data/diabetes_limpio.csv"
RUTA_DATOS_NUEVOS     = "data/nuevos/"
RUTA_DATOS_UNIFICADOS = "data/datos_unificados.csv"
RUTA_MODELO           = "models/mejor_modelo.pkl"
RUTA_COLUMNAS         = "models/columnas.pkl"
RUTA_UMBRAL           = "models/umbral.pkl"
RUTA_METRICAS         = "models/metricas.txt"
 
# Ratio desbalance — igual que en entrenar.py
SCALE_POS_WEIGHT = 8
 
 
def hay_datos_nuevos():
    """Revisa si hay CSVs en la carpeta de datos nuevos."""
    if not os.path.exists(RUTA_DATOS_NUEVOS):
        return False
    return len([f for f in os.listdir(RUTA_DATOS_NUEVOS) if f.endswith(".csv")]) > 0
 
 
def cargar_datos_nuevos():
    """Une todos los CSVs de datos nuevos en un solo DataFrame."""
    archivos = [f for f in os.listdir(RUTA_DATOS_NUEVOS) if f.endswith(".csv")]
    dfs = []
    for archivo in archivos:
        ruta = os.path.join(RUTA_DATOS_NUEVOS, archivo)
        df   = pd.read_csv(ruta)
        dfs.append(df)
        print(f"  -> Cargado: {archivo} ({len(df):,} filas)")
    return pd.concat(dfs, ignore_index=True)
 
 
def leer_metricas_actuales():
    """Lee las metricas del modelo en produccion."""
    metricas = {"auc_test": 0.0, "recall_clase1": 0.0, "f1_clase1": 0.0, "umbral": 0.5}
    if not os.path.exists(RUTA_METRICAS):
        return metricas
    try:
        with open(RUTA_METRICAS, "r") as f:
            for linea in f:
                if "=" in linea:
                    clave, valor = linea.strip().split("=")
                    if clave in metricas:
                        metricas[clave] = float(valor)
    except Exception:
        pass
    return metricas
 
 
def optimizar_umbral(modelo, X_test, y_test):
    """Busca el umbral que maximiza F1 de la clase positiva."""
    y_proba      = modelo.predict_proba(X_test)[:, 1]
    mejor_f1     = 0
    mejor_umbral = 0.5
    for umbral in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= umbral).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > mejor_f1:
            mejor_f1     = f1
            mejor_umbral = umbral
    return mejor_umbral
 
 
def entrenar_con_datos(df):
    """
    Reentrena XGBoost con los parametros del modelo actual.
    Si no existe modelo previo, usa parametros razonables por defecto.
    """
    X = df.drop(columns=["readmitido_30dias"])
    y = df["readmitido_30dias"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    # Cargamos los parametros del modelo actual si existe
    if os.path.exists(RUTA_MODELO):
        modelo_actual = joblib.load(RUTA_MODELO)
        # XGBoost guarda los parametros en get_params()
        params_actuales = modelo_actual.get_params()
        params = {
            "n_estimators":     params_actuales.get("n_estimators", 200),
            "max_depth":        params_actuales.get("max_depth", 5),
            "learning_rate":    params_actuales.get("learning_rate", 0.1),
            "subsample":        params_actuales.get("subsample", 0.8),
            "colsample_bytree": params_actuales.get("colsample_bytree", 0.8),
            "min_child_weight": params_actuales.get("min_child_weight", 3),
            "gamma":            params_actuales.get("gamma", 0),
            "reg_alpha":        params_actuales.get("reg_alpha", 0),
            "reg_lambda":       params_actuales.get("reg_lambda", 1),
        }
    else:
        params = {
            "n_estimators": 200, "max_depth": 5,
            "learning_rate": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 3,
        }
 
    modelo_nuevo = XGBClassifier(
        **params,
        scale_pos_weight = SCALE_POS_WEIGHT,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
        eval_metric      = "auc",
    )
    modelo_nuevo.fit(X_train, y_train)
 
    # Optimizamos umbral
    umbral   = optimizar_umbral(modelo_nuevo, X_test, y_test)
    y_proba  = modelo_nuevo.predict_proba(X_test)[:, 1]
    y_pred   = (y_proba >= umbral).astype(int)
 
    metricas = {
        "auc_test":      round(roc_auc_score(y_test, y_proba), 4),
        "recall_clase1": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_clase1":     round(f1_score(y_test, y_pred, zero_division=0), 4),
        "umbral":        round(umbral, 2),
    }
 
    return modelo_nuevo, metricas, X_train, y_test, y_pred
 
 
def reentrenar():
    """
    Funcion principal.
    Revisa datos nuevos, reentrena si los hay,
    y solo reemplaza el modelo si el nuevo es mejor.
    """
    print("=" * 55)
    print("REENTRENAMIENTO AUTOMATICO")
    print("=" * 55)
 
    if not hay_datos_nuevos():
        print("No hay datos nuevos en data/nuevos/. No se reentrena.")
        return
 
    print("Datos nuevos encontrados!")
 
    # Unir datos nuevos con existentes
    df_nuevos   = cargar_datos_nuevos()
    df_actuales = pd.read_csv(RUTA_DATOS_ACTUALES)
    df_total    = pd.concat([df_actuales, df_nuevos], ignore_index=True)
    df_total.to_csv(RUTA_DATOS_UNIFICADOS, index=False)
    print(f"Total datos para entrenamiento: {len(df_total):,} filas")
 
    # Entrenar modelo nuevo
    print("\nEntrenando modelo nuevo con XGBoost...")
    modelo_nuevo, metricas_nuevas, X_train, y_test, y_pred = entrenar_con_datos(df_total)
 
    # Comparar con modelo actual
    metricas_actuales = leer_metricas_actuales()
    print(f"\nAUC modelo actual:  {metricas_actuales['auc_test']:.4f}")
    print(f"AUC modelo nuevo:   {metricas_nuevas['auc_test']:.4f}")
    print(f"Recall actual:      {metricas_actuales['recall_clase1']:.4f}")
    print(f"Recall nuevo:       {metricas_nuevas['recall_clase1']:.4f}")
 
    if metricas_nuevas["auc_test"] > metricas_actuales["auc_test"]:
        print("\nEl modelo nuevo es mejor. Reemplazando...")
 
        # Guardar nuevo modelo, columnas y umbral
        os.makedirs("models", exist_ok=True)
        joblib.dump(modelo_nuevo, RUTA_MODELO)
        joblib.dump(list(X_train.columns), RUTA_COLUMNAS)
        joblib.dump(metricas_nuevas["umbral"], RUTA_UMBRAL)
 
        with open(RUTA_METRICAS, "w") as f:
            for k, v in metricas_nuevas.items():
                f.write(f"{k}={v}\n")
 
        # Registrar en MLflow
        mlflow.set_experiment("readmision-hospitalaria-xgboost")
        with mlflow.start_run(run_name="reentrenamiento-automatico"):
            mlflow.log_metric("auc_test",        metricas_nuevas["auc_test"])
            mlflow.log_metric("auc_previo",       metricas_actuales["auc_test"])
            mlflow.log_metric("recall_clase1",    metricas_nuevas["recall_clase1"])
            mlflow.log_metric("f1_clase1",        metricas_nuevas["f1_clase1"])
            mlflow.log_metric("datos_nuevos",     len(df_nuevos))
            mlflow.log_param("umbral_decision",   metricas_nuevas["umbral"])
            mlflow.sklearn.log_model(modelo_nuevo, "modelo")
 
            # Matriz de confusion
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred,
                display_labels=["No readmitido", "Readmitido"],
                cmap="Blues", ax=ax
            )
            ax.set_title("Matriz de Confusion - Reentrenamiento")
            fig.savefig("models/confusion_matrix_reentrenamiento.png", bbox_inches="tight")
            mlflow.log_artifact("models/confusion_matrix_reentrenamiento.png")
            plt.close(fig)
 
        print("Modelo actualizado y registrado en MLflow!")
 
        # Limpiar carpeta de datos nuevos
        for archivo in os.listdir(RUTA_DATOS_NUEVOS):
            if archivo.endswith(".csv"):
                os.remove(os.path.join(RUTA_DATOS_NUEVOS, archivo))
        print("Carpeta data/nuevos/ limpiada.")
 
    else:
        print("\nEl modelo actual es mejor. No se reemplaza.")
        print("Los datos nuevos se conservan para el proximo reentrenamiento.")
 
    print("=" * 55)
 
 
if __name__ == "__main__":
    reentrenar()