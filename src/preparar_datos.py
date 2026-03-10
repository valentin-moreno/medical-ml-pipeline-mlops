# Paso 1: Cargar y limpiar los datos
# Dataset: Diabetes 130-US Hospitals (UCI Machine Learning Repository)
# Lo que hace este script:
#   - Descarga el dataset si no existe localmente
#   - Limpia valores faltantes y columnas inútiles
#   - Crea la columna objetivo: ¿el paciente fue readmitido en menos de 30 dias?
#   - Guarda el resultado listo para entrenar
#
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA

import pandas as pd
import numpy as np
import os

# Ruta donde vamos a guardar los datos procesados
RUTA_DATOS_CRUDOS  = "data/diabetes_raw.csv"
RUTA_DATOS_LIMPIOS = "data/diabetes_limpio.csv"

# URL del dataset publico de UCI
URL_DATASET = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"


def cargar_datos():
    """
    Carga el dataset desde disco.
    Si no existe, avisa al usuario que lo descargue.
    Retorna un DataFrame de pandas.
    """
    if not os.path.exists(RUTA_DATOS_CRUDOS):
        print("No se encontro el archivo de datos.")
        print(f"Descargalo desde: {URL_DATASET}")
        print(f"y guardalo como: {RUTA_DATOS_CRUDOS}")
        raise FileNotFoundError(f"Falta el archivo: {RUTA_DATOS_CRUDOS}")

    print("Cargando datos...")
    df = pd.read_csv(RUTA_DATOS_CRUDOS)
    print(f"  -> {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def limpiar_datos(df):
    """
    Limpia el DataFrame crudo.
    - Reemplaza '?' por NaN (valor nulo de pandas)
    - Elimina columnas que no aportan informacion util
    - Elimina filas con demasiados valores faltantes
    Retorna el DataFrame limpio.
    """
    print("Limpiando datos...")

    # El dataset usa '?' para representar valores desconocidos
    # Los reemplazamos por NaN para que pandas los maneje bien
    df = df.replace("?", np.nan)

    # Estas columnas tienen demasiados nulos o no aportan al modelo
    columnas_a_eliminar = [
        "encounter_id",       # solo es un ID, no tiene valor predictivo
        "patient_nbr",        # mismo caso, ID del paciente
        "weight",             # mas del 96% son nulos
        "payer_code",         # informacion de seguro, no clinica
        "medical_specialty",  # muchos nulos
    ]

    # Eliminamos solo las que existen en el DataFrame
    # (por si el dataset cambia de version)
    columnas_a_eliminar = [c for c in columnas_a_eliminar if c in df.columns]
    df = df.drop(columns=columnas_a_eliminar)

    # Eliminamos filas donde mas del 40% de los valores son nulos
    umbral_nulos = 0.4
    df = df.dropna(thresh=int(df.shape[1] * (1 - umbral_nulos)))

    print(f"  -> Despues de limpiar: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def crear_variable_objetivo(df):
    """
    Crea la columna 'readmitido_30dias' que es lo que queremos predecir.
    
    El dataset original tiene tres valores en 'readmitted':
      - '<30'  -> fue readmitido en menos de 30 dias (esto es lo que nos preocupa)
      - '>30'  -> fue readmitido despues de 30 dias
      - 'NO'   -> no fue readmitido
    
    Nosotros simplificamos: 1 si fue readmitido en menos de 30 dias, 0 si no.
    Retorna el DataFrame con la nueva columna.
    """
    print("Creando variable objetivo...")

    # 1 si readmitido en menos de 30 dias, 0 en cualquier otro caso
    df["readmitido_30dias"] = (df["readmitted"] == "<30").astype(int)

    # Ya no necesitamos la columna original
    df = df.drop(columns=["readmitted"])

    # Mostramos cuantos pacientes fueron readmitidos
    n_readmitidos = df["readmitido_30dias"].sum()
    pct = n_readmitidos / len(df) * 100
    print(f"  -> Readmitidos en <30 dias: {n_readmitidos} ({pct:.1f}%)")

    return df


def codificar_categoricas(df):
    """
    Convierte columnas de texto en numeros para que el modelo las entienda.
    Usamos Label Encoding simple — suficiente para este proyecto.
    Retorna el DataFrame con columnas numericas.
    """
    print("Codificando variables categoricas...")

    # Identificamos columnas de tipo texto (object en pandas)
    columnas_texto = df.select_dtypes(include=["object"]).columns.tolist()

    for columna in columnas_texto:
        # Convertimos cada categoria unica a un numero entero
        # Ejemplo: 'Male' -> 0, 'Female' -> 1
        df[columna] = pd.Categorical(df[columna]).codes

    print(f"  -> {len(columnas_texto)} columnas codificadas")
    return df


def guardar_datos(df):
    """Guarda el DataFrame limpio en disco."""
    os.makedirs("data", exist_ok=True)
    df.to_csv(RUTA_DATOS_LIMPIOS, index=False)
    print(f"Datos guardados en: {RUTA_DATOS_LIMPIOS}")


def preparar():
    """
    Funcion principal.
    Ejecuta todos los pasos en orden y retorna el DataFrame listo.
    """
    print("=" * 50)
    print("PREPARACION DE DATOS")
    print("=" * 50)

    df = cargar_datos()
    df = limpiar_datos(df)
    df = crear_variable_objetivo(df)
    df = codificar_categoricas(df)
    guardar_datos(df)

    print("=" * 50)
    print("Datos listos para entrenar!")
    print("=" * 50)

    return df


# Si corres este archivo directamente, ejecuta la preparacion
if __name__ == "__main__":
    preparar()
