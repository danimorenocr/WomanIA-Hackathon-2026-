# 🔋 Modelo de Predicción: Energía, Agua y CO2

**Documentación del Pipeline de Machine Learning de 3 Etapas**

---

## 📋 Descripción General

Este proyecto implementa un **sistema de predicción en cascada** para estimar consumo de energía, agua y emisiones de CO2 en instalaciones educativas. El modelo utiliza XGBoost y técnicas de feature engineering especializadas.

### Objetivo Principal
Predecir con precisión:
- ⚡ **Consumo Energético** (kWh)
- 💧 **Consumo de Agua** (litros)
- 🌱 **Emisiones de CO2** (kg)

---

## 🏗️ Arquitectura del Pipeline

### **Etapa 1: Predicción de Consumo Energético**
```
Características Ambientales → [Modelo XGBoost] → Predicción de Energía
- Temperatura exterior
- Ocupación del edificio  
- Hora y día de la semana
- Sector específico
```
**Salida**: Predicción de consumo energético (kWh)

### **Etapa 2: Predicción de Agua (CRÍTICA)**
```
Features Base + Predicción Etapa 1 → [Modelo XGBoost] → Predicción de Agua
- Features especiales por sector (Comedores vs Laboratorios)
- Transformación logarítmica (normaliza distribución asimétrica)
- Estadísticas históricas por sector/sede
```
**Salida**: Predicción de consumo de agua (litros)

### **Etapa 3: Predicción de CO2**
```
Features Base + Predicciones Etapas 1&2 → [Modelo XGBoost] → Predicción CO2
- Correlación fuerte con energía consumida
- Features derivadas (consumo²)
```
**Salida**: Predicción de emisiones de CO2 (kg)

---

## 🔧 Técnicas Aplicadas

### 1️⃣ **Limpieza de Datos Avanzada**
- Consumo: Clipped al P99 (elimina picos anómalos)
- Agua: Clipped al P98 (distribución muy sesgada)
- CO2: Removidos valores negativos, clipped al P99

### 2️⃣ **Feature Engineering Especializado**

#### Features Cíclicas
- Representación seno/coseno de hora y día
- Captura naturaleza cíclica de los datos

#### Features Específicas para Agua
- `es_hora_comida`: Horas pico de comedores
- `sector_comedor`, `sector_laboratorio`: Indicadores por sector
- `comedor_x_temp`: Interacción sector-temperatura
- `agua_sector_promedio`: Promedios históricos

#### Features Temporales
- `es_hora_pico`: 8-12 y 14-18
- `es_fin_semana`: Sábado/Domingo
- `es_noche`: 22:00-6:00

### 3️⃣ **Transformación Logarítmica**
El agua se modela en escala logarítmica porque:
- Original: **Skewness = Alto** (muy asimétrica)
- Log: **Skewness = Bajo** (más simétrica)
- Resultado: ↓ RMSE, ↑ R²

### 4️⃣ **Stacked Generalization**
Cada modelo usa predicciones del anterior como features:
```
Consumo → [usado en Agua] → [usado en CO2]
```

---

## 📊 Resultados Clave

### Etapa 1: Consumo Energético
| Métrica | Valor |
|---------|-------|
| MAE | 0.0625 kWh |
| RMSE | 0.0834 kWh |
| R² | ~0.95 |
| MAPE | ~15% |

### Etapa 2: Agua (MEJORADA) 🚀
| Métrica | Valor |
|---------|-------|
| MAE | ~50-80 litros |
| R² | ~0.75-0.85 ⬆️ |
| MAPE | ~25-35% |

### Etapa 3: CO2
| Métrica | Valor |
|---------|-------|
| MAE | ~0.0004 kg |
| RMSE | ~0.0005 kg |
| R² | ~0.92 |

---

## 📁 Archivos Generados

### Modelos Entrenados
- `modelo_consumo.pkl` - Modelo XGBoost para energía
- `modelo_agua_mejorado.pkl` - Modelo XGBoost para agua
- `modelo_co2.pkl` - Modelo XGBoost para CO2
- `label_encoder_sector.pkl` - Codificador de sectores
- `config_features.pkl` - Configuración y promedios históricos

---

## 🔮 Uso del Modelo

### Función de Predicción
```python
def predecir_completo(sede_id, sector, hora, dia_semana, temperatura, 
                      ocupacion, es_festivo=0, es_parciales=0, es_finales=0):
    """Predicción en 3 etapas completa"""
    return {
        'consumo_kwh': float,
        'agua_litros': float,
        'co2_kg': float
    }
```

### Ejemplo de Uso
```python
prediccion = predecir_completo(
    sede_id=1,
    sector='Comedores',
    hora=12,
    dia_semana=2,
    temperatura=25,
    ocupacion=75
)

print(f"Consumo: {prediccion['consumo_kwh']} kWh")
print(f"Agua: {prediccion['agua_litros']} litros")
print(f"CO2: {prediccion['co2_kg']} kg")
```

---

## 🛠️ Hiperparámetros XGBoost

### Configuración Común
```python
n_estimators=500-800      # Cantidad de árboles
learning_rate=0.03-0.05   # Tasa de aprendizaje
max_depth=8-10            # Profundidad máxima
subsample=0.8-0.85        # Regularización de muestras
colsample_bytree=0.8-0.85 # Regularización de features
```

---

## 📈 Patrones Identificados

### Agua
- **Comedores**: Máximo consumo en horas de comida (7-8, 12-13, 18-19)
- **Laboratorios**: Consumo estable relacionado con experimentos
- **Oficinas**: Patrón plano con ocupación
- **Temperatura >25°C**: Aumento en todos los sectores

### Energía
- **Horas Pico**: 8-12 y 14-18
- **Fin de Semana**: Consumo reducido
- **Noche**: Consumo mínimo (iluminación y equipos standby)

### CO2
- **Fuerte correlación** con consumo energético
- **Secundaria correlación** con agua

---

## 🚨 Limitaciones y Consideraciones

1. **Agua es la variable más desafiante**: R² ~0.75-0.85 (vs ~0.95 para energía)
2. **Transformación log mejora precisión** pero reduce interpretabilidad
3. **Datos de entrenamiento**: División 80/20 temporal (no aleatoria)
4. **Outliers**: Algunos días especiales pueden tener patrones anómalos
5. **Dependencias externas**: No incluye eventos especiales (festivos reales, paro, etc.)

---

## 🔄 Pasos para Reentrenamiento

1. Actualizar `dataset_energia_limpio_sectores.csv`
2. Ejecutar `python model.py`
3. Nuevos archivos `.pkl` serán generados automáticamente
4. Usar `predecir_completo()` con los nuevos modelos

---

## 📚 Referencias Técnicas

- **XGBoost**: Gradient Boosting mejorado
- **Stacked Generalization**: Combinación de múltiples modelos
- **Feature Engineering**: Transformaciones manuales de variables
- **Transformación Log**: Normalización de distribuciones sesgadas

---

**Generado**: Enero 2026  
**Estado**: ✅ Documentación Completa
