# Tests basicos del pipeline
# Verifican que las funciones principales funcionan correctamente
# Correr con: pytest tests/
#
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA

import sys
import os
import pandas as pd
import numpy as np

# Para que Python encuentre los modulos en src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preparar_datos import (
    limpiar_datos,
    crear_variable_objetivo,
    codificar_categoricas,
)


def crear_df_prueba():
    """
    Crea un DataFrame pequeño para usar en los tests.
    No necesitamos el dataset real para probar la logica.
    """
    return pd.DataFrame({
        "encounter_id":    [1, 2, 3, 4, 5],
        "patient_nbr":     [100, 101, 102, 103, 104],
        "age":             ["[50-60)", "[60-70)", "[40-50)", "?", "[70-80)"],
        "gender":          ["Male", "Female", "Male", "Female", "Male"],
        "weight":          ["?", "?", "?", "?", "?"],          # columna a eliminar
        "time_in_hospital":[3, 7, 2, 5, 1],
        "num_medications": [10, 15, 8, 12, 6],
        "readmitted":      ["<30", "NO", ">30", "<30", "NO"],  # variable objetivo
    })


def test_limpieza_elimina_columnas():
    """Verifica que se eliminan las columnas inutiles."""
    df = crear_df_prueba()
    df_limpio = limpiar_datos(df)

    # Estas columnas deben desaparecer
    assert "encounter_id" not in df_limpio.columns
    assert "patient_nbr"  not in df_limpio.columns
    assert "weight"        not in df_limpio.columns


def test_limpieza_reemplaza_interrogacion():
    """Verifica que los '?' se convierten en NaN."""
    df = crear_df_prueba()
    df_limpio = limpiar_datos(df)

    # No debe quedar ningun '?' en el DataFrame
    tiene_interrogacion = (df_limpio == "?").any().any()
    assert not tiene_interrogacion


def test_variable_objetivo_es_binaria():
    """Verifica que la columna objetivo solo tiene 0 y 1."""
    df = crear_df_prueba()
    df = limpiar_datos(df)
    df = crear_variable_objetivo(df)

    valores_unicos = df["readmitido_30dias"].unique()
    assert set(valores_unicos).issubset({0, 1})


def test_variable_objetivo_correcta():
    """Verifica que '<30' se mapea a 1 y el resto a 0."""
    df = crear_df_prueba()
    df = limpiar_datos(df)
    df = crear_variable_objetivo(df)

    # El dataset tiene 2 filas con '<30', deben ser 2 unos
    assert df["readmitido_30dias"].sum() == 2


def test_columna_readmitted_desaparece():
    """Verifica que la columna original 'readmitted' se elimina."""
    df = crear_df_prueba()
    df = limpiar_datos(df)
    df = crear_variable_objetivo(df)

    assert "readmitted" not in df.columns


def test_codificacion_no_deja_strings():
    """Verifica que despues de codificar no quedan columnas de texto."""
    df = crear_df_prueba()
    df = limpiar_datos(df)
    df = crear_variable_objetivo(df)
    df = codificar_categoricas(df)

    # No debe quedar ninguna columna de tipo object (texto)
    columnas_texto = df.select_dtypes(include=["object"]).columns.tolist()
    assert len(columnas_texto) == 0


if __name__ == "__main__":
    tests = [
        test_limpieza_elimina_columnas,
        test_limpieza_reemplaza_interrogacion,
        test_variable_objetivo_es_binaria,
        test_variable_objetivo_correcta,
        test_columna_readmitted_desaparece,
        test_codificacion_no_deja_strings,
    ]

    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
