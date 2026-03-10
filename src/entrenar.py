# Paso 2: Entrenar el modelo
# Lo que hace este script:
#   - Carga los datos limpios
#   - Usa Optuna para encontrar los mejores hiperparametros automaticamente
#   - Registra cada experimento en MLflow (para no perder nada)
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Rutas importantes
RUTA_DATOS  = "data/diabetes_limpio.csv"
RUTA_MODELO = "models/mejor_modelo.pkl"

# Nombre del experimento en MLflow
# Asi podemos ver todos los entrenamientos juntos en la UI
NOMBRE_EXPERIMENTO = "readmision-hospitalaria"

# Cuantas combinaciones de hiperparametros va a probar Optuna
# Mas trials = mejor modelo pero mas tiempo
# Para el portafolio con 30 esta bien
N_TRIALS = 30


def cargar_datos():
    """
    Carga el dataset limpio y lo separa en features (X) y objetivo (y).
    Retorna X_train, X_test, y_train, y_test.
    """
    print("Cargando datos limpios...")

    if not os.path.exists(RUTA_DATOS):
        raise FileNotFoundError(
            f"No se encontro {RUTA_DATOS}. "
            "Corre primero: python src/preparar_datos.py"
        )

    df = pd.read_csv(RUTA_DATOS)

    # Separamos lo que queremos predecir (y) del resto (X)
    X = df.drop(columns=["readmitido_30dias"])
    y = df["readmitido_30dias"]

    # 80% para entrenar, 20% para evaluar
    # random_state=42 para que siempre sea la misma division (reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  -> Entrenamiento: {X_train.shape[0]} muestras")
    print(f"  -> Prueba:        {X_test.shape[0]} muestras")
    print(f"  -> Features:      {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def objetivo_optuna(trial, X_train, y_train):
    """
    Esta funcion es la que Optuna llama en cada trial.
    Optuna le pasa un objeto 'trial' que sugiere valores de hiperparametros.
    Nosotros entrenamos el modelo con esos valores y retornamos el AUC-ROC.
    Optuna aprende que valores funcionan mejor y los mejora en cada trial.
    """

    # Optuna sugiere valores dentro de los rangos que le damos
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth    = trial.suggest_int("max_depth", 3, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10)

    # Creamos el modelo con los hiperparametros sugeridos
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Validacion cruzada con 3 folds — mas confiable que un solo split
    scores = cross_val_score(
        modelo, X_train, y_train,
        cv=3, scoring="roc_auc", n_jobs=-1
    )

    # Retornamos el promedio — Optuna intentara maximizar este valor
    return scores.mean()


def entrenar(X_train, X_test, y_train, y_test):
    """
    Ejecuta la busqueda de hiperparametros con Optuna
    y registra todo en MLflow.
    Retorna el mejor modelo entrenado.
    """

    # Configuramos MLflow
    mlflow.set_experiment(NOMBRE_EXPERIMENTO)

    print(f"\nBuscando mejores hiperparametros ({N_TRIALS} intentos)...")
    print("Esto puede tomar unos minutos...\n")

    # Optuna minimiza por defecto, le decimos que maximize el AUC
    estudio = optuna.create_study(direction="maximize")

    # Le pasamos X_train y y_train a la funcion objetivo via lambda
    estudio.optimize(
        lambda trial: objetivo_optuna(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    # Sacamos los mejores hiperparametros que encontro Optuna
    mejores_params = estudio.best_params
    mejor_auc_cv   = estudio.best_value

    print(f"\nMejores hiperparametros encontrados:")
    for param, valor in mejores_params.items():
        print(f"  {param}: {valor}")
    print(f"\nMejor AUC-ROC en validacion cruzada: {mejor_auc_cv:.4f}")

    # Entrenamos el modelo final con los mejores parametros
    # Esta vez usamos TODO el conjunto de entrenamiento
    print("\nEntrenando modelo final...")
    mejor_modelo = RandomForestClassifier(
        **mejores_params,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    mejor_modelo.fit(X_train, y_train)

    # Evaluamos en el conjunto de prueba (datos que el modelo nunca vio)
    y_pred      = mejor_modelo.predict(X_test)
    y_pred_proba = mejor_modelo.predict_proba(X_test)[:, 1]
    auc_test    = roc_auc_score(y_test, y_pred_proba)

    print(f"\nResultados en conjunto de prueba:")
    print(f"  AUC-ROC: {auc_test:.4f}")
    print(f"\nReporte de clasificacion:")
    print(classification_report(y_test, y_pred))

    # Registramos todo en MLflow para no perder nada
    # Despues podemos ver esto en la UI con: mlflow ui
    with mlflow.start_run(run_name="mejor-modelo"):

        # Guardamos los hiperparametros
        mlflow.log_params(mejores_params)

        # Guardamos las metricas
        mlflow.log_metric("auc_cv",   mejor_auc_cv)
        mlflow.log_metric("auc_test", auc_test)

        # Guardamos el modelo directamente en MLflow
        mlflow.sklearn.log_model(mejor_modelo, "modelo")

        print(f"\nExperimento registrado en MLflow!")
        print("Para ver la UI corre: mlflow ui")

    return mejor_modelo, auc_test, mejores_params


def guardar_modelo(modelo, X_train):
    """Guarda el modelo y los nombres de columnas en disco."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, RUTA_MODELO)
    # Guardamos los nombres de columnas para que la API los use
    joblib.dump(list(X_train.columns), "models/columnas.pkl")
    print(f"\nModelo guardado en: {RUTA_MODELO}")


def entrenar_pipeline():
    """
    Funcion principal.
    Ejecuta todo el flujo de entrenamiento de principio a fin.
    """
    print("=" * 50)
    print("ENTRENAMIENTO DEL MODELO")
    print("=" * 50)

    X_train, X_test, y_train, y_test = cargar_datos()
    modelo, auc, params = entrenar(X_train, X_test, y_train, y_test)
    guardar_modelo(modelo, X_train)

    print("\n" + "=" * 50)
    print(f"Entrenamiento completado!")
    print(f"AUC-ROC final: {auc:.4f}")
    print("=" * 50)

    return modelo


# Si corres este archivo directamente, ejecuta el entrenamiento
if __name__ == "__main__":
    entrenar_pipeline()
