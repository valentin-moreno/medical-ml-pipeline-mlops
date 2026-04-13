# Paso 1: Preparacion de datos con feature engineering
# Dataset: Diabetes 130-US Hospitals (UCI Machine Learning Repository)
# Lo que hace este script:
#   - Limpia valores faltantes y columnas inutiles
#   - Convierte age de rango de texto a numero real
#   - Agrupa diagnosticos CIE-9 en categorias clinicas
#   - Codifica medicamentos como ordinal (No=0, Steady=1, Up/Down=2)
#   - Crea features clinicos nuevos que capturan riesgo de readmision
#   - Crea la variable objetivo: readmitido en menos de 30 dias
#
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA

import pandas as pd
import numpy as np
import os

RUTA_DATOS_CRUDOS  = "data/diabetes_raw.csv"
RUTA_DATOS_LIMPIOS = "data/diabetes_limpio.csv"


def cargar_datos():
    """Carga el dataset crudo desde disco."""
    if not os.path.exists(RUTA_DATOS_CRUDOS):
        raise FileNotFoundError(
            f"No se encontro {RUTA_DATOS_CRUDOS}.\n"
            "Descargalo desde: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008\n"
            "y guardalo como: data/diabetes_raw.csv"
        )
    print("Cargando datos...")
    df = pd.read_csv(RUTA_DATOS_CRUDOS)
    print(f"  -> {df.shape[0]:,} filas, {df.shape[1]} columnas")
    return df


def limpiar_basico(df):
    """
    Limpieza inicial.
    Reemplaza '?' por NaN y elimina columnas sin valor predictivo.
    weight tiene 96% de nulos. examide y citoglipton tienen un solo valor.
    """
    print("Limpieza basica...")

    df = df.replace("?", np.nan)

    columnas_eliminar = [
        "encounter_id", "patient_nbr",
        "weight",
        "payer_code",
        "medical_specialty",
        "examide",
        "citoglipton",
    ]
    df = df.drop(columns=[c for c in columnas_eliminar if c in df.columns])

    filas_antes = len(df)
    df = df.dropna(thresh=int(df.shape[1] * 0.6))
    print(f"  -> Filas eliminadas por nulos: {filas_antes - len(df):,}")
    print(f"  -> Resultado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    return df


def convertir_age(df):
    """
    Convierte age de rango de texto a punto medio numerico.
    '[70-80)' -> 75. Evita orden artificial del Label Encoding en rangos.
    """
    print("Convirtiendo age a numero...")

    mapeo_age = {
        "[0-10)":   5,  "[10-20)":  15, "[20-30)":  25,
        "[30-40)":  35, "[40-50)":  45, "[50-60)":  55,
        "[60-70)":  65, "[70-80)":  75, "[80-90)":  85,
        "[90-100)": 95,
    }
    df["age"] = df["age"].map(mapeo_age).fillna(df["age"].map(mapeo_age).median())
    return df


def agrupar_diagnosticos(df):
    """
    Agrupa codigos CIE-9 en categorias clinicas en vez de Label Encoding.
    Label Encoding de codigos como 250.01, 428.0 introduce orden numerico
    sin sentido clinico — el modelo interpretaria 428 > 250 erroneamente.
    """
    print("Agrupando diagnosticos CIE-9 en categorias clinicas...")

    def codigo_a_categoria(codigo):
        if pd.isna(codigo):
            return 0
        codigo = str(codigo).strip()
        if codigo.startswith("E") or codigo.startswith("V"):
            return 8
        try:
            num = float(codigo)
        except ValueError:
            return 0

        if 390 <= num <= 459 or num == 785:  return 1  # cardiovascular
        elif 250 <= num <= 250.99:           return 2  # diabetes
        elif 460 <= num <= 519 or num == 786:return 3  # respiratorio
        elif 520 <= num <= 579 or num == 787:return 4  # digestivo
        elif 800 <= num <= 999:              return 5  # trauma
        elif 710 <= num <= 739:              return 6  # musculoesqueletico
        elif 580 <= num <= 629 or num == 788:return 7  # genitourinario
        else:                                return 9  # otros

    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].apply(codigo_a_categoria)

    return df


def codificar_medicamentos(df):
    """
    Codifica medicamentos como ordinal clinicamente significativo.
    No=0 (sin uso), Steady=1 (dosis estable), Up/Down=2 (dosis cambiada).
    Los cambios de dosis son relevantes para predecir readmision.
    """
    print("Codificando medicamentos como ordinal clinico...")

    columnas_medicamentos = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "insulin",
        "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]

    mapeo_med = {"No": 0, "Steady": 1, "Down": 2, "Up": 2}
    for col in columnas_medicamentos:
        if col in df.columns:
            df[col] = df[col].map(mapeo_med).fillna(0).astype(int)

    return df


def codificar_binarias(df):
    """Codifica variables categoricas simples con mapeo explicito."""
    print("Codificando variables categoricas simples...")

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1}).fillna(0).astype(int)

    if "change" in df.columns:
        df["change"] = df["change"].map({"No": 0, "Ch": 1}).fillna(0).astype(int)

    if "diabetesMed" in df.columns:
        df["diabetesMed"] = df["diabetesMed"].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    if "max_glu_serum" in df.columns:
        df["max_glu_serum"] = df["max_glu_serum"].map(
            {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
        ).fillna(0).astype(int)

    if "A1Cresult" in df.columns:
        df["A1Cresult"] = df["A1Cresult"].map(
            {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
        ).fillna(0).astype(int)

    if "race" in df.columns:
        df["race"] = pd.Categorical(df["race"]).codes

    return df


def crear_features(df):
    """
    Feature engineering clinico.
    Crea variables nuevas que capturan patrones de riesgo que las
    variables originales no expresan directamente.
    """
    print("Creando features clinicos...")

    columnas_medicamentos = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "insulin",
        "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]

    # Total de visitas previas — mas visitas = mayor riesgo de readmision
    cols_visitas = [c for c in ["number_outpatient", "number_emergency", "number_inpatient"] if c in df.columns]
    if cols_visitas:
        df["total_visitas_previas"] = df[cols_visitas].sum(axis=1)

    # Complejidad clinica del paciente
    if all(c in df.columns for c in ["num_procedures", "num_medications", "number_diagnoses"]):
        df["complejidad_clinica"] = (
            df["num_procedures"] +
            df["num_medications"] / 10 +
            df["number_diagnoses"]
        )

    # Uso de medicamentos antidiabeticos
    cols_med = [c for c in columnas_medicamentos if c in df.columns]
    if cols_med:
        df["medicamentos_activos"]  = (df[cols_med] > 0).sum(axis=1)
        df["medicamentos_cambiados"] = (df[cols_med] == 2).sum(axis=1)

    # Intensidad de procedimientos por dia de estancia
    if all(c in df.columns for c in ["num_lab_procedures", "time_in_hospital"]):
        df["intensidad_hospitalaria"] = (
            df["num_lab_procedures"] / df["time_in_hospital"].replace(0, 1)
        )

    # Diagnostico principal es diabetes
    if "diag_1" in df.columns:
        df["diag_principal_diabetes"] = (df["diag_1"] == 2).astype(int)

    return df


def crear_variable_objetivo(df):
    """Crea la variable objetivo: 1 si readmitido en <30 dias, 0 si no."""
    print("Creando variable objetivo...")

    df["readmitido_30dias"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    n   = df["readmitido_30dias"].sum()
    pct = n / len(df) * 100
    print(f"  -> Readmitidos en <30 dias: {n:,} ({pct:.1f}%)")
    print(f"  -> No readmitidos:          {len(df) - n:,} ({100-pct:.1f}%)")
    return df


def guardar_datos(df):
    """Guarda el DataFrame preparado en disco."""
    os.makedirs("data", exist_ok=True)
    df.to_csv(RUTA_DATOS_LIMPIOS, index=False)
    print(f"Datos guardados en: {RUTA_DATOS_LIMPIOS}")
    print(f"  -> {df.shape[0]:,} filas, {df.shape[1]} columnas finales")


def preparar():
    """Ejecuta el pipeline completo de preparacion de datos."""
    print("=" * 55)
    print("PREPARACION DE DATOS CON FEATURE ENGINEERING")
    print("=" * 55)

    df = cargar_datos()
    df = limpiar_basico(df)
    df = convertir_age(df)
    df = agrupar_diagnosticos(df)
    df = codificar_medicamentos(df)
    df = codificar_binarias(df)
    df = crear_features(df)
    df = crear_variable_objetivo(df)
    guardar_datos(df)

    print("=" * 55)
    print("Datos listos para entrenar!")
    print("=" * 55)
    return df


if __name__ == "__main__":
    preparar()