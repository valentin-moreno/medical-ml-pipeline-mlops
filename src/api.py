# Paso 3: Servir el modelo como API REST
# Lo que hace este script:
#   - Carga el modelo XGBoost y el umbral optimizado desde disco
#   - Expone endpoint para hacer predicciones con todos los features
#   - Retorna probabilidad, riesgo y umbral usado
#
# Valentin Moreno Vasquez - Ingeniero Biomedico, Especialista en IA
 
import joblib
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
 
RUTA_MODELO   = "models/mejor_modelo.pkl"
RUTA_COLUMNAS = "models/columnas.pkl"
RUTA_UMBRAL   = "models/umbral.pkl"
 
app = FastAPI(
    title       = "Prediccion de Readmision Hospitalaria",
    description = (
        "Predice si un paciente diabetico sera readmitido en menos de 30 dias. "
        "Modelo: XGBoost optimizado con Optuna + umbral de decision optimizado. "
        "MLOps: experimentos trackeados con MLflow."
    ),
    version="2.0.0",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Estado global — se carga una vez al arrancar
modelo   = None
columnas = None
umbral   = 0.5
 
 
@app.on_event("startup")
def cargar_modelo():
    """Carga el modelo, columnas y umbral al arrancar la API."""
    global modelo, columnas, umbral
 
    if not os.path.exists(RUTA_MODELO):
        print(f"ADVERTENCIA: No se encontro el modelo en {RUTA_MODELO}")
        print("Corre primero: python src/entrenar.py")
        return
 
    modelo   = joblib.load(RUTA_MODELO)
    columnas = joblib.load(RUTA_COLUMNAS)
 
    if os.path.exists(RUTA_UMBRAL):
        umbral = joblib.load(RUTA_UMBRAL)
 
    print(f"Modelo cargado. Umbral de decision: {umbral:.2f}")
 
 
class DatosPaciente(BaseModel):
    """Datos clinicos del paciente para la prediccion."""
 
    # Demograficos
    edad:                         int            # edad en anos (ej: 75)
    genero:                       int            # 0: masculino, 1: femenino
 
    # Hospitalizacion
    tiempo_internacion:           int            # dias en hospital
    numero_procedimientos:        int
    numero_medicamentos:          int
    numero_diagnosticos:          int
 
    # Visitas previas
    numero_consultas_emergencia:  int
    numero_hospitalizaciones:     int
    numero_consultas_ambulatorias:int
 
    # Diabetes
    resultado_hba1c:              Optional[int] = 0   # 0: no medido, 1: normal, 2: >7, 3: >8
    resultado_glucosa:            Optional[int] = 0   # 0: no medido, 1: normal, 2: >200, 3: >300
    cambio_medicamento:           Optional[int] = 0   # 0: no, 1: si
    medicamento_diabetes:         Optional[int] = 1   # 0: no, 1: si
    insulina:                     Optional[int] = 0   # 0: no, 1: estable, 2: cambiada
    metformina:                   Optional[int] = 0   # 0: no, 1: estable, 2: cambiada
 
    # ID opcional
    paciente_id: Optional[str] = None
 
 
@app.get("/")
def raiz():
    return {
        "mensaje": "API de Readmision Hospitalaria activa",
        "modelo":  "XGBoost + Optuna",
        "umbral":  umbral,
        "docs":    "/docs"
    }
 
 
@app.get("/health")
def health():
    return {
        "status":          "ok" if modelo else "modelo no cargado",
        "umbral_decision": umbral,
    }
 
 
@app.post("/predecir")
def predecir(datos: DatosPaciente):
    """
    Recibe datos clinicos del paciente y retorna la prediccion de readmision.
    Usa el umbral optimizado durante el entrenamiento.
    """
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Corre python src/entrenar.py")
 
    try:
        # Construimos el DataFrame con todas las columnas del modelo
        # inicializadas en 0 — luego llenamos las que llegaron
        fila = pd.DataFrame([{col: 0 for col in columnas}])
 
        # Mapeamos los campos del request a las columnas del modelo
        mapeo = {
            "age":                          datos.edad,
            "gender":                       datos.genero,
            "time_in_hospital":             datos.tiempo_internacion,
            "num_procedures":               datos.numero_procedimientos,
            "num_medications":              datos.numero_medicamentos,
            "number_diagnoses":             datos.numero_diagnosticos,
            "number_emergency":             datos.numero_consultas_emergencia,
            "number_inpatient":             datos.numero_hospitalizaciones,
            "number_outpatient":            datos.numero_consultas_ambulatorias,
            "A1Cresult":                    datos.resultado_hba1c,
            "max_glu_serum":               datos.resultado_glucosa,
            "change":                       datos.cambio_medicamento,
            "diabetesMed":                  datos.medicamento_diabetes,
            "insulin":                      datos.insulina,
            "metformin":                    datos.metformina,
        }
 
        for col, valor in mapeo.items():
            if col in fila.columns:
                fila[col] = valor
 
        # Recalculamos features derivados
        if "total_visitas_previas" in fila.columns:
            fila["total_visitas_previas"] = (
                datos.numero_consultas_emergencia +
                datos.numero_hospitalizaciones +
                datos.numero_consultas_ambulatorias
            )
        if "complejidad_clinica" in fila.columns:
            fila["complejidad_clinica"] = (
                datos.numero_procedimientos +
                datos.numero_medicamentos / 10 +
                datos.numero_diagnosticos
            )
        if "medicamentos_activos" in fila.columns:
            fila["medicamentos_activos"] = (datos.insulina > 0) + (datos.metformina > 0)
        if "medicamentos_cambiados" in fila.columns:
            fila["medicamentos_cambiados"] = (datos.insulina == 2) + (datos.metformina == 2)
        if "intensidad_hospitalaria" in fila.columns:
            fila["intensidad_hospitalaria"] = 0
 
        # Prediccion con umbral optimizado
        probabilidad = float(modelo.predict_proba(fila)[0][1])
        readmision   = probabilidad >= umbral
 
        if probabilidad >= 0.5:
            nivel_riesgo = "ALTO"
        elif probabilidad >= 0.3:
            nivel_riesgo = "MEDIO"
        else:
            nivel_riesgo = "BAJO"
 
        return {
            "paciente_id":     datos.paciente_id,
            "probabilidad":    round(probabilidad, 4),
            "readmision_30d":  bool(readmision),
            "riesgo":          nivel_riesgo,
            "umbral_usado":    round(umbral, 2),
            "interpretacion":  (
                "El modelo predice que este paciente SERA readmitido en menos de 30 dias."
                if readmision else
                "El modelo predice que este paciente NO sera readmitido en menos de 30 dias."
            ),
        }
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 