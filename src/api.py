# Paso 3: Servir el modelo como API
# Lo que hace este script:
#   - Carga el modelo entrenado desde disco
#   - Expone un endpoint para hacer predicciones
#   - Cualquier sistema puede preguntarle: "este paciente se va a readmitir?"
#
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA

import joblib
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Ruta del modelo guardado por el script de entrenamiento
RUTA_MODELO = "models/mejor_modelo.pkl"

# Iniciamos la aplicacion FastAPI
app = FastAPI(
    title="Prediccion de Readmision Hospitalaria",
    description=(
        "Predice si un paciente diabetico sera readmitido en menos de 30 dias. "
        "Modelo: Random Forest optimizado con Optuna. "
        "MLOps: experimentos trackeados con MLflow."
    ),
    version="1.0.0",
)

# Permitir peticiones desde cualquier origen (util para demos y frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargamos el modelo una sola vez al iniciar la API
# (no en cada peticion — eso seria muy lento)
modelo = None

@app.on_event("startup")
def cargar_modelo():
    """Se ejecuta automaticamente cuando la API arranca."""
    global modelo
    if not os.path.exists(RUTA_MODELO):
        print(f"ADVERTENCIA: No se encontro el modelo en {RUTA_MODELO}")
        print("Corre primero: python src/entrenar.py")
        return
    modelo = joblib.load(RUTA_MODELO)
    print("Modelo cargado correctamente!")


# Modelos de datos 
# Pydantic valida automaticamente que los datos tengan el tipo correcto
# Si algo falta o tiene tipo incorrecto, FastAPI retorna un error claro

class DatosPaciente(BaseModel):
    """Datos clinicos del paciente para hacer la prediccion."""

    # Datos demograficos
    edad: int                          # rango de edad en decadas (ej: 50-60 -> 5)
    genero: int                        # 0: femenino, 1: masculino

    # Datos de la hospitalizacion
    tiempo_internacion: int            # dias en el hospital
    numero_procedimientos: int         # procedimientos realizados
    numero_medicamentos: int           # medicamentos administrados
    numero_diagnosticos: int           # diagnosticos registrados
    numero_consultas_emergencia: int   # visitas a emergencia en el ultimo ano
    numero_hospitalizaciones: int      # hospitalizaciones en el ultimo ano
    numero_consultas_ambulatorias: int # consultas ambulatorias en el ultimo ano

    # Variables clinicas de diabetes
    resultado_hba1c: Optional[int] = 0    # resultado HbA1c (0: no medido, 1-4: rango)
    cambio_medicamento: Optional[int] = 0  # 0: no hubo cambio, 1: si hubo cambio
    medicamento_diabetes: Optional[int] = 1 # 0: no, 1: si recibe medicamento

    # ID opcional para identificar la prediccion
    paciente_id: Optional[str] = None


class ResultadoPrediccion(BaseModel):
    """Resultado de la prediccion."""
    paciente_id: Optional[str]
    readmision_en_30_dias: bool   # True si el modelo predice readmision
    probabilidad: float            # probabilidad entre 0 y 1
    riesgo: str                    # 'BAJO', 'MEDIO' o 'ALTO'
    mensaje: str                   # explicacion en texto


# Endpoints 

@app.get("/")
def raiz():
    return {
        "mensaje": "API de Prediccion de Readmision Hospitalaria 🏥",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    """Verifica que la API y el modelo estan listos."""
    return {
        "status": "ok",
        "modelo_cargado": modelo is not None
    }


@app.post("/predecir", response_model=ResultadoPrediccion)
def predecir(datos: DatosPaciente):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    # Cargamos las columnas con las que se entrenó el modelo
    columnas = joblib.load("models/columnas.pkl")

    # Creamos un DataFrame con todas las columnas en cero
    datos_df = pd.DataFrame([{col: 0 for col in columnas}])

    # Llenamos solo las columnas que nos mandaron
    datos_df["age"]               = datos.edad
    datos_df["gender"]            = datos.genero
    datos_df["time_in_hospital"]  = datos.tiempo_internacion
    datos_df["num_procedures"]    = datos.numero_procedimientos
    datos_df["num_medications"]   = datos.numero_medicamentos
    datos_df["number_diagnoses"]  = datos.numero_diagnosticos
    datos_df["number_emergency"]  = datos.numero_consultas_emergencia
    datos_df["number_inpatient"]  = datos.numero_hospitalizaciones
    datos_df["number_outpatient"] = datos.numero_consultas_ambulatorias
    datos_df["A1Cresult"]         = datos.resultado_hba1c
    datos_df["change"]            = datos.cambio_medicamento
    datos_df["diabetesMed"]       = datos.medicamento_diabetes

    probabilidad  = modelo.predict_proba(datos_df)[0][1]
    prediccion    = probabilidad >= 0.5

    if probabilidad < 0.3:
        riesgo  = "BAJO"
        mensaje = "Baja probabilidad de readmision. Control ambulatorio recomendado."
    elif probabilidad < 0.6:
        riesgo  = "MEDIO"
        mensaje = "Riesgo moderado. Se recomienda seguimiento cercano al alta."
    else:
        riesgo  = "ALTO"
        mensaje = "Alto riesgo de readmision. Considerar extension de hospitalizacion o protocolo de seguimiento intensivo."

    return ResultadoPrediccion(
        paciente_id=datos.paciente_id,
        readmision_en_30_dias=prediccion,
        probabilidad=round(float(probabilidad), 4),
        riesgo=riesgo,
        mensaje=mensaje,
    )


@app.get("/modelo/info")
def info_modelo():
    """Retorna informacion basica del modelo cargado."""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    return {
        "tipo": type(modelo).__name__,
        "n_estimators": modelo.n_estimators,
        "max_depth": modelo.max_depth,
        "n_features": modelo.n_features_in_,
    }
