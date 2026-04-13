# Medical ML Pipeline 🏥

**Valentín Moreno Vásquez** · Biomedical Engineer · AI Specialist

---

## ¿Qué hace este proyecto? / What does this do?

**ES:** Pipeline completo de Machine Learning que predice si un paciente diabético será readmitido en menos de 30 días. Incluye optimización automática de hiperparámetros con Optuna, seguimiento de experimentos con MLflow, API REST para hacer predicciones en tiempo real, y reentrenamiento automático cuando llegan datos nuevos.

**EN:** End-to-end ML pipeline that predicts whether a diabetic patient will be readmitted within 30 days. Features automatic hyperparameter optimization with Optuna, experiment tracking with MLflow, a REST API for real-time predictions, and automatic retraining when new data arrives.

---

## Stack

- **Python 3.10+**
- **scikit-learn** — modelo Random Forest
- **Optuna** — optimización automática de hiperparámetros
- **MLflow** — seguimiento de experimentos y registro de modelos
- **FastAPI** — API REST para predicciones
- **Pytest** — tests automáticos

---

## Arquitectura / Architecture

```
data/             ← datos crudos y procesados
src/
  preparar_datos.py  ← limpieza y transformacion del dataset
  entrenar.py        ← entrenamiento + Optuna + MLflow
  api.py             ← API REST para predicciones
  reentrenar.py      ← reentrenamiento automatico con datos nuevos
models/           ← modelo entrenado guardado en disco
tests/            ← tests automaticos
```

---

## Dataset

**Diabetes 130-US Hospitals** (UCI Machine Learning Repository)
- 100,000+ registros de hospitalizaciones reales
- Descarga: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
- Guardar como: `data/diabetes_raw.csv`

---

## Cómo correrlo / How to run it

```bash
# 1. Clonar el repositorio
git clone https://github.com/valentin-moreno/medical-ml-pipeline.git
cd medical-ml-pipeline

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el dataset y guardarlo en data/diabetes_raw.csv

# 5. Limpiar los datos
python src/preparar_datos.py

# 6. Entrenar el modelo (Optuna + MLflow)
python src/entrenar.py

# 7. Ver experimentos en MLflow
mlflow ui
# Abrir: http://localhost:5000

# 8. Levantar la API
uvicorn src.api:app --reload
# Abrir: http://localhost:8000/docs
```

---

## Ejemplo de prediccion / Prediction example

```bash
curl -X POST "http://localhost:8000/predecir" \
  -H "Content-Type: application/json" \
  -d '{
    "paciente_id": "p_001",
    "edad": 5,
    "genero": 1,
    "tiempo_internacion": 7,
    "numero_procedimientos": 3,
    "numero_medicamentos": 15,
    "numero_diagnosticos": 7,
    "numero_consultas_emergencia": 2,
    "numero_hospitalizaciones": 1,
    "numero_consultas_ambulatorias": 0
  }'
```

**Respuesta:**
```json
{
  "paciente_id": "p_001",
  "readmision_en_30_dias": true,
  "probabilidad": 0.6821,
  "riesgo": "ALTO",
  "mensaje": "Alto riesgo de readmision. Considerar extension de hospitalizacion o protocolo de seguimiento intensivo."
}
```

---

## Reentrenamiento automático / Automatic retraining

Cuando llegan datos nuevos, solo hay que copiarlos a `data/nuevos/` y correr:

```bash
python src/reentrenar.py
```

El script solo reemplaza el modelo si el nuevo es mejor que el actual.

---

## Tests

```bash
pytest tests/
```

---

## Roadmap

- [ ] Agregar monitoreo de data drift con Evidently
- [ ] Containerizar con Docker
- [ ] Deploy en Render o Railway

---

## Autor / Author

**Valentín Moreno Vásquez**
Biomedical Engineer · AI Specialist
[LinkedIn](#) · [GitHub](#)
