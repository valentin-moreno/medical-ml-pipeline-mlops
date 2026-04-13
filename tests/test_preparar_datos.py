# Tests del pipeline de preparacion de datos
# Verifican limpieza, feature engineering y variable objetivo
# Correr con: pytest tests/ -v
#
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA
 
import sys
import os
import pandas as pd
import numpy as np
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
from src.preparar_datos import (
    limpiar_basico,
    convertir_age,
    agrupar_diagnosticos,
    codificar_medicamentos,
    codificar_binarias,
    crear_features,
    crear_variable_objetivo,
)
 
 
def crear_df_prueba():
    """
    DataFrame minimo que simula la estructura del dataset real.
    Suficiente para probar toda la logica sin necesitar el CSV real.
    """
    return pd.DataFrame({
        "encounter_id":              [1, 2, 3, 4, 5],
        "patient_nbr":               [100, 101, 102, 103, 104],
        "age":                       ["[50-60)", "[60-70)", "[40-50)", "[70-80)", "[80-90)"],
        "gender":                    ["Male", "Female", "Male", "Female", "Male"],
        "race":                      ["White", "Black", "Hispanic", "White", "Asian"],
        "weight":                    ["?", "?", "?", "?", "?"],
        "payer_code":                ["MC", "?", "HM", "BC", "?"],
        "medical_specialty":         ["Internal", "?", "Cardiology", "?", "Surgery"],
        "examide":                   ["No", "No", "No", "No", "No"],
        "citoglipton":               ["No", "No", "No", "No", "No"],
        "time_in_hospital":          [3, 7, 2, 5, 1],
        "num_lab_procedures":        [40, 60, 30, 55, 20],
        "num_procedures":            [1, 3, 0, 2, 1],
        "num_medications":           [10, 15, 8, 12, 6],
        "number_outpatient":         [0, 1, 0, 2, 0],
        "number_emergency":          [0, 0, 1, 0, 0],
        "number_inpatient":          [1, 2, 0, 1, 0],
        "number_diagnoses":          [5, 8, 3, 7, 4],
        "diag_1":                    ["250.01", "428.0", "486", "410.71", "250.00"],
        "diag_2":                    ["401.9", "250.0", "?", "428.0", "585.9"],
        "diag_3":                    ["272.4", "?", "496", "?", "401.9"],
        "max_glu_serum":             ["None", ">200", "None", ">300", "Norm"],
        "A1Cresult":                 ["None", ">7", ">8", "None", "Norm"],
        "metformin":                 ["Steady", "No", "Up", "Steady", "No"],
        "repaglinide":               ["No", "No", "No", "No", "No"],
        "nateglinide":               ["No", "No", "No", "No", "No"],
        "chlorpropamide":            ["No", "No", "No", "No", "No"],
        "glimepiride":               ["No", "No", "No", "No", "No"],
        "acetohexamide":             ["No", "No", "No", "No", "No"],
        "glipizide":                 ["No", "No", "No", "No", "No"],
        "glyburide":                 ["No", "No", "No", "No", "No"],
        "tolbutamide":               ["No", "No", "No", "No", "No"],
        "pioglitazone":              ["No", "No", "No", "No", "No"],
        "rosiglitazone":             ["No", "No", "No", "No", "No"],
        "acarbose":                  ["No", "No", "No", "No", "No"],
        "miglitol":                  ["No", "No", "No", "No", "No"],
        "troglitazone":              ["No", "No", "No", "No", "No"],
        "tolazamide":                ["No", "No", "No", "No", "No"],
        "insulin":                   ["Steady", "Up", "No", "Down", "Steady"],
        "glyburide-metformin":       ["No", "No", "No", "No", "No"],
        "glipizide-metformin":       ["No", "No", "No", "No", "No"],
        "glimepiride-pioglitazone":  ["No", "No", "No", "No", "No"],
        "metformin-rosiglitazone":   ["No", "No", "No", "No", "No"],
        "metformin-pioglitazone":    ["No", "No", "No", "No", "No"],
        "change":                    ["Ch", "No", "Ch", "No", "Ch"],
        "diabetesMed":               ["Yes", "Yes", "No", "Yes", "Yes"],
        "readmitted":                ["<30", "NO", ">30", "<30", "NO"],
    })
 
 
def test_limpieza_elimina_columnas_inutiles():
    """Verifica que se eliminan columnas sin valor predictivo."""
    df = crear_df_prueba()
    df = df.replace("?", np.nan)
    df_limpio = limpiar_basico(df.copy())
 
    for col in ["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty", "examide", "citoglipton"]:
        assert col not in df_limpio.columns, f"La columna {col} deberia eliminarse"
 
 
def test_limpieza_reemplaza_interrogacion():
    """Verifica que no quedan '?' en el DataFrame."""
    df = crear_df_prueba()
    df_limpio = limpiar_basico(df.copy())
    assert not (df_limpio == "?").any().any()
 
 
def test_age_se_convierte_a_numero():
    """Verifica que age pasa de rango de texto a numero entero."""
    df = crear_df_prueba()
    df = limpiar_basico(df.replace("?", np.nan))
    df = convertir_age(df)
    assert pd.api.types.is_numeric_dtype(df["age"])
    assert df["age"].max() <= 100
    assert df["age"].min() >= 0
 
 
def test_age_valores_correctos():
    """Verifica que los rangos se convierten al punto medio correcto."""
    df = pd.DataFrame({"age": ["[50-60)", "[70-80)", "[0-10)"]})
    df = convertir_age(df)
    assert df["age"].tolist() == [55, 75, 5]
 
 
def test_diagnosticos_agrupados_en_categorias():
    """Verifica que los codigos CIE-9 se agrupan en categorias numericas."""
    df = crear_df_prueba()
    df = limpiar_basico(df.replace("?", np.nan))
    df = convertir_age(df)
    df = agrupar_diagnosticos(df)
    # Todas las categorias deben ser enteros entre 0 y 9
    for col in ["diag_1", "diag_2", "diag_3"]:
        assert df[col].between(0, 9).all(), f"{col} tiene valores fuera de rango"
 
 
def test_diabetes_mapeada_correctamente():
    """Verifica que 250.x se mapea a categoria 2 (diabetes)."""
    df = pd.DataFrame({"diag_1": ["250.01", "250.00", "428.0"]})
    df = agrupar_diagnosticos(df)
    assert df["diag_1"].iloc[0] == 2
    assert df["diag_1"].iloc[1] == 2
    assert df["diag_1"].iloc[2] == 1  # cardiovascular
 
 
def test_medicamentos_codificados_como_ordinal():
    """Verifica que medicamentos se convierten a 0, 1, 2."""
    df = crear_df_prueba()
    df = limpiar_basico(df.replace("?", np.nan))
    df = convertir_age(df)
    df = agrupar_diagnosticos(df)
    df = codificar_medicamentos(df)
    assert df["metformin"].isin([0, 1, 2]).all()
    assert df["insulin"].isin([0, 1, 2]).all()
 
 
def test_medicamento_no_es_cero():
    """Verifica que 'No' se mapea a 0."""
    df = pd.DataFrame({"metformin": ["No", "Steady", "Up", "Down"]})
    df = codificar_medicamentos(df)
    assert df["metformin"].tolist() == [0, 1, 2, 2]
 
 
def test_features_creados():
    """Verifica que el feature engineering crea las columnas esperadas."""
    df = crear_df_prueba()
    df = limpiar_basico(df.replace("?", np.nan))
    df = convertir_age(df)
    df = agrupar_diagnosticos(df)
    df = codificar_medicamentos(df)
    df = codificar_binarias(df)
    df = crear_features(df)
 
    features_esperados = [
        "total_visitas_previas",
        "complejidad_clinica",
        "medicamentos_activos",
        "medicamentos_cambiados",
        "intensidad_hospitalaria",
        "diag_principal_diabetes",
    ]
    for feature in features_esperados:
        assert feature in df.columns, f"Falta el feature: {feature}"
 
 
def test_total_visitas_es_suma_correcta():
    """Verifica que total_visitas_previas es la suma de las 3 columnas."""
    df = crear_df_prueba()
    df = limpiar_basico(df.replace("?", np.nan))
    df = convertir_age(df)
    df = agrupar_diagnosticos(df)
    df = codificar_medicamentos(df)
    df = codificar_binarias(df)
    df = crear_features(df)
 
    esperado = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    pd.testing.assert_series_equal(df["total_visitas_previas"], esperado, check_names=False)
 
 
def test_variable_objetivo_es_binaria():
    """Verifica que la variable objetivo solo tiene 0 y 1."""
    df = crear_df_prueba()
    df = limpiar_basico(df.replace("?", np.nan))
    df = crear_variable_objetivo(df)
    assert set(df["readmitido_30dias"].unique()).issubset({0, 1})
 
 
def test_variable_objetivo_correcta():
    """Verifica que solo '<30' se mapea a 1."""
    df = crear_df_prueba()
    df = limpiar_basico(df.replace("?", np.nan))
    df = crear_variable_objetivo(df)
    # El df de prueba tiene 2 filas con '<30'
    assert df["readmitido_30dias"].sum() == 2
 
 
def test_columna_readmitted_desaparece():
    """Verifica que la columna original readmitted se elimina."""
    df = crear_df_prueba()
    df = limpiar_basico(df.replace("?", np.nan))
    df = crear_variable_objetivo(df)
    assert "readmitted" not in df.columns
 
 
def test_no_quedan_columnas_texto():
    """Verifica que despues del pipeline completo no quedan strings."""
    df = crear_df_prueba()
    df = limpiar_basico(df.replace("?", np.nan))
    df = convertir_age(df)
    df = agrupar_diagnosticos(df)
    df = codificar_medicamentos(df)
    df = codificar_binarias(df)
    df = crear_features(df)
    df = crear_variable_objetivo(df)
 
    columnas_texto = df.select_dtypes(include=["object"]).columns.tolist()
    assert len(columnas_texto) == 0, f"Quedan columnas texto: {columnas_texto}"
 
 
if __name__ == "__main__":
    tests = [
        test_limpieza_elimina_columnas_inutiles,
        test_limpieza_reemplaza_interrogacion,
        test_age_se_convierte_a_numero,
        test_age_valores_correctos,
        test_diagnosticos_agrupados_en_categorias,
        test_diabetes_mapeada_correctamente,
        test_medicamentos_codificados_como_ordinal,
        test_medicamento_no_es_cero,
        test_features_creados,
        test_total_visitas_es_suma_correcta,
        test_variable_objetivo_es_binaria,
        test_variable_objetivo_correcta,
        test_columna_readmitted_desaparece,
        test_no_quedan_columnas_texto,
    ]
 
    for test in tests:
        try:
            test()
            print(f"  OK {test.__name__}")
        except Exception as e:
            print(f"  FAIL {test.__name__}: {e}")
 