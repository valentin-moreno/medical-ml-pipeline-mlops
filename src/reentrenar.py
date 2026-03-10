# Automatizacion: Reentrenamiento automatico
# Lo que hace este script:
#   - Revisa si llegaron datos nuevos a la carpeta data/nuevos/
#   - Si hay datos nuevos, los une con los existentes y reentrena el modelo
#   - Registra el nuevo entrenamiento en MLflow
#   - Solo reentrena si los datos nuevos mejoran el modelo
#
# Para automatizarlo puedes usar cron (Linux/Mac) o Task Scheduler (Windows):
#   cron ejemplo: 0 2 * * * python src/reentrenar.py  (corre a las 2am cada dia)
#
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA

import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Rutas del proyecto
RUTA_DATOS_ACTUALES = "data/diabetes_limpio.csv"
RUTA_DATOS_NUEVOS   = "data/nuevos/"           # carpeta donde llegan datos nuevos
RUTA_DATOS_UNIFICADOS = "data/datos_unificados.csv"
RUTA_MODELO         = "models/mejor_modelo.pkl"
RUTA_METRICAS       = "models/metricas.txt"    # guardamos el AUC del modelo actual


def hay_datos_nuevos():
    """
    Revisa si existe la carpeta de datos nuevos y tiene archivos CSV.
    Retorna True si hay datos para procesar, False si no.
    """
    if not os.path.exists(RUTA_DATOS_NUEVOS):
        return False

    # Buscamos archivos CSV en la carpeta
    archivos_csv = [
        f for f in os.listdir(RUTA_DATOS_NUEVOS)
        if f.endswith(".csv")
    ]

    return len(archivos_csv) > 0


def cargar_datos_nuevos():
    """
    Une todos los CSV de la carpeta de datos nuevos en un solo DataFrame.
    Retorna el DataFrame combinado.
    """
    archivos = [
        f for f in os.listdir(RUTA_DATOS_NUEVOS)
        if f.endswith(".csv")
    ]

    dfs = []
    for archivo in archivos:
        ruta = os.path.join(RUTA_DATOS_NUEVOS, archivo)
        df = pd.read_csv(ruta)
        dfs.append(df)
        print(f"  -> Cargado: {archivo} ({len(df)} filas)")

    return pd.concat(dfs, ignore_index=True)


def leer_auc_actual():
    """
    Lee el AUC del modelo actualmente en produccion.
    Si no existe el archivo de metricas, asume AUC = 0.
    """
    if not os.path.exists(RUTA_METRICAS):
        return 0.0

    with open(RUTA_METRICAS, "r") as f:
        linea = f.read().strip()

    # El archivo guarda algo como: "auc_test=0.6823"
    try:
        auc = float(linea.split("=")[1])
        return auc
    except Exception:
        return 0.0


def guardar_auc(auc):
    """Guarda el AUC del modelo en disco para comparar en el futuro."""
    os.makedirs("models", exist_ok=True)
    with open(RUTA_METRICAS, "w") as f:
        f.write(f"auc_test={auc:.4f}")


def entrenar_con_datos(df):
    """
    Entrena un RandomForest con los parametros del modelo actual.
    Retorna el modelo entrenado y su AUC en el conjunto de prueba.
    """
    X = df.drop(columns=["readmitido_30dias"])
    y = df["readmitido_30dias"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Usamos los parametros del modelo actual si existe
    # Si no, usamos valores razonables por defecto
    if os.path.exists(RUTA_MODELO):
        modelo_actual = joblib.load(RUTA_MODELO)
        params = {
            "n_estimators":     modelo_actual.n_estimators,
            "max_depth":        modelo_actual.max_depth,
            "min_samples_split": modelo_actual.min_samples_split,
            "min_samples_leaf": modelo_actual.min_samples_leaf,
        }
    else:
        params = {
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        }

    modelo_nuevo = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    modelo_nuevo.fit(X_train, y_train)

    y_proba = modelo_nuevo.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    return modelo_nuevo, auc


def reentrenar():
    """
    Funcion principal.
    Revisa si hay datos nuevos, reentrena si los hay,
    y solo reemplaza el modelo si el nuevo es mejor.
    """
    print("=" * 50)
    print("REVISION DE DATOS NUEVOS")
    print("=" * 50)

    # Paso 1: verificar si hay datos nuevos
    if not hay_datos_nuevos():
        print("No hay datos nuevos. No se reentrena.")
        return

    print("Datos nuevos encontrados!")

    # Paso 2: cargar y unir datos nuevos con los existentes
    df_nuevos   = cargar_datos_nuevos()
    df_actuales = pd.read_csv(RUTA_DATOS_ACTUALES)
    df_total    = pd.concat([df_actuales, df_nuevos], ignore_index=True)
    df_total.to_csv(RUTA_DATOS_UNIFICADOS, index=False)
    print(f"Total de datos para entrenamiento: {len(df_total)} filas")

    # Paso 3: entrenar modelo nuevo con todos los datos
    print("\nEntrenando modelo nuevo...")
    modelo_nuevo, auc_nuevo = entrenar_con_datos(df_total)

    # Paso 4: comparar con el modelo actual
    auc_actual = leer_auc_actual()
    print(f"\nAUC modelo actual: {auc_actual:.4f}")
    print(f"AUC modelo nuevo:  {auc_nuevo:.4f}")

    # Paso 5: solo reemplazamos si el nuevo es mejor (aunque sea un poco)
    if auc_nuevo > auc_actual:
        print("\nEl modelo nuevo es mejor! Reemplazando...")

        # Guardamos el nuevo modelo
        joblib.dump(modelo_nuevo, RUTA_MODELO)
        guardar_auc(auc_nuevo)

        # Registramos en MLflow
        mlflow.set_experiment("readmision-hospitalaria")
        with mlflow.start_run(run_name="reentrenamiento-automatico"):
            mlflow.log_metric("auc_test",  auc_nuevo)
            mlflow.log_metric("auc_previo", auc_actual)
            mlflow.log_metric("datos_nuevos", len(df_nuevos))
            mlflow.sklearn.log_model(modelo_nuevo, "modelo")

        print("Modelo actualizado y registrado en MLflow!")

        # Limpiamos la carpeta de datos nuevos (ya los procesamos)
        for archivo in os.listdir(RUTA_DATOS_NUEVOS):
            if archivo.endswith(".csv"):
                os.remove(os.path.join(RUTA_DATOS_NUEVOS, archivo))
        print("Carpeta de datos nuevos limpiada.")

    else:
        print("\nEl modelo actual es mejor. No se reemplaza.")
        print("Los datos nuevos se conservan para el proximo reentrenamiento.")

    print("=" * 50)


if __name__ == "__main__":
    reentrenar()
