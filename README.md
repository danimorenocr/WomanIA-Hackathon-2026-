# 🌿 API de Análisis Energético Universitario

Sistema de predicción y análisis energético para universidades usando Machine Learning y LLM.

## 📁 Estructura del Proyecto

```
WomanIA-Hackathon-2026/
├── api.py                    # API Flask principal
├── llm_engine.py             # Motor de predicción ML
├── generar_graficos.py       # Generador de datos para gráficos
├── preguntas_predefinidas.py # Sistema de preguntas naturales
├── requirements.txt          # Dependencias Python
├── .env                      # Variables de entorno (API keys)
├── data/
│   └── dataset_energia_limpio_sectores.csv
└── models/
    ├── modelo_consumo.pkl
    ├── modelo_agua_mejorado.pkl
    ├── modelo_co2.pkl
    └── config_features.pkl
```

## 🚀 Instalación

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## ▶️ Ejecutar

```bash
python api.py
```

Servidor: `http://localhost:5000`

---

## 📡 ENDPOINTS DE LA API

### 🤖 Chat con IA (Groq - GRATIS)

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/chat-groq?mensaje=Hola` | Chat con query param |
| POST | `/api/chat-groq` | Chat con body JSON |

**Ejemplo POST:**
```json
{"mensaje": "¿Cómo puedo ahorrar energía?"}
```

**Respuesta:**
```json
{
    "respuesta": "¡Hola! Para ahorrar energía te recomiendo...",
    "modelo": "qwen/qwen3-32b"
}
```

---

### 📊 Gráficos y Análisis

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/graficos` | **Todos los gráficos en JSON** |
| GET | `/api/consumo-por-sede` | Consumo total por sede |
| GET | `/api/tendencias-consumo` | Tendencias últimos 30 días |
| GET | `/api/eficiencia-estudiante` | Eficiencia por estudiante |
| GET | `/api/emisiones-co2` | Emisiones CO₂ por sede |
| GET | `/api/consumo-agua` | Consumo de agua por sede |
| GET | `/api/temperatura-consumo` | Temperatura vs consumo |
| GET | `/api/consumo-por-sector` | Consumo por sector |
| GET | `/api/distribucion-por-sector` | Distribución % por sector |
| GET | `/api/tendencias-sector` | Tendencias por sector |
| GET | `/api/eficiencia-sector-sede` | Eficiencia sector × sede |
| GET | `/api/correlacion-ocupacion` | Ocupación vs consumo |
| GET | `/api/costos-operacionales` | Costos COP por sector |
| GET | `/api/impacto-ambiental` | CO₂ + agua + árboles |

---

### 🔮 Predicción ML

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/predecir` | Predicción completa |

**Body:**
```json
{
    "sede_id": 2,
    "sector": "Laboratorios",
    "hora": 14,
    "dia_semana": 2,
    "temperatura": 25,
    "ocupacion": 75
}
```

**Respuesta:**
```json
{
    "energia_kwh": 1250.45,
    "agua_litros": 3500.20,
    "co2_kg": 425.30,
    "sede": "Tunja",
    "sector": "Laboratorios"
}
```

---

### 💬 Preguntas Naturales

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/preguntas` | Lista de preguntas predefinidas |
| POST | `/api/responder-pregunta` | Responder por ID |
| POST | `/api/chat` | Chat para predicciones |

**Ejemplo:**
```json
{"pregunta": "¿Cuánta energía consumirá el laboratorio de la sede 3 mañana a las 5pm?"}
```

---

## 🏢 Sedes

| ID | Nombre |
|----|--------|
| 1 | Chiquinquirá |
| 2 | Tunja |
| 3 | Duitama |
| 4 | Sogamoso |

## 🏗️ Sectores

- Comedores
- Salones
- Laboratorios
- Auditorios
- Oficinas

---

## 🔑 Configuración (.env)

```env
GROQ_API_KEY=tu-groq-api-key-aqui
```

Obtén tu API key gratis: https://console.groq.com/keys

## 📊 Tecnologías

- **Backend**: Flask + Flask-CORS
- **ML**: XGBoost, Scikit-learn
- **LLM**: Groq (Qwen3-32B) - Gratis
- **Data**: Pandas, NumPy
