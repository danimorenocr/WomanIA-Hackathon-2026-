# 🌿 API de Análisis Energético Universitario

Sistema de predicción y análisis energético para universidades usando Machine Learning y LLM.

## 📁 Estructura del Proyecto

```
WomanIA-Hackathon-2026/
├── api.py                    # API Flask principal (único punto de entrada)
├── llm_engine.py             # Motor de predicción ML (XGBoost)
├── generar_graficos.py       # Generador de datos para gráficos
├── preguntas_predefinidas.py # Sistema de preguntas naturales
├── requirements.txt          # Dependencias Python
├── .env                      # Variables de entorno (API keys)
├── data/
│   └── dataset_energia_limpio_sectores.csv  # Datos históricos
└── models/
    ├── modelo_consumo.pkl    # Modelo energía (XGBoost)
    ├── modelo_agua_mejorado.pkl # Modelo agua
    ├── modelo_co2.pkl        # Modelo emisiones CO₂
    └── config_features.pkl   # Configuración de features
```

## 🚀 Instalación

```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar API key de Groq (gratis)
# Editar .env y agregar: GROQ_API_KEY=tu-api-key
```

## ▶️ Ejecutar

```bash
python api.py
```

Servidor disponible en: `http://localhost:5000`

## 📡 Endpoints

### Chat con IA (Groq - GRATIS)
```
GET  /api/chat-groq?mensaje=Hola
POST /api/chat-groq  {"mensaje": "¿Cómo ahorrar energía?"}
```

### Predicción ML
```
POST /api/predecir
{
    "sede_id": 2,
    "sector": "Laboratorios",
    "hora": 14,
    "dia_semana": 2,
    "temperatura": 25,
    "ocupacion": 75
}
```

### Gráficos y Análisis
| Endpoint | Descripción |
|----------|-------------|
| `/api/graficos` | Todos los gráficos |
| `/api/consumo-por-sede` | Consumo por sede |
| `/api/consumo-por-sector` | Consumo por sector |
| `/api/emisiones-co2` | Emisiones CO₂ |
| `/api/costos-operacionales` | Costos COP |

### Preguntas Naturales
```
POST /api/chat
{"pregunta": "¿Cuánta energía consumirá el laboratorio de la sede 3 mañana a las 5pm?"}
```

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

## 🔑 Configuración (.env)

```env
GROQ_API_KEY=tu-groq-api-key-aqui
```

Obtén tu API key gratis en: https://console.groq.com/keys

## 📊 Tecnologías

- **Backend**: Flask + Flask-CORS
- **ML**: XGBoost, Scikit-learn
- **LLM**: Groq (Qwen3-32B) - Gratis
- **Data**: Pandas, NumPy
