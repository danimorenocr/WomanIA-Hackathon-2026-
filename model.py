"""
MODELO DE PREDICCIÓN DE CONSUMO ENERGÉTICO, AGUA Y CO2
========================================================

Este módulo implementa un pipeline de machine learning en 3 etapas para predecir:
1. Consumo energético (kWh) por sector
2. Consumo de agua (litros) 
3. Emisiones de CO2 (kg)

Basado en datos de temperatura, ocupación, hora, día y características de instalaciones
educativas. Utiliza XGBoost para regresión con features engineered especializadas.

Autor(as): [Karol Acuña, Daniela Moreno, Sofia Torres, Juliana Garzón]
Fecha: Enero 31 / 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ================================================================
# PREPROCESAMIENTO BASE
# ================================================================

df = pd.read_csv('dataset_energia_limpio_sectores.csv')

# ================================================================
# AJUSTE DE ESCALAS A VALORES REALES (CRÍTICO)
# ================================================================
# Los datos del dataset vienen escalados/normalizados. Esta sección
# reconvierte todos los valores a sus unidades reales para un análisis
# más interpretable y precisión en las predicciones.

# Energía: viene en miles → pasar a kWh reales
energia_cols = [
    'energia_total_kwh',
    'energia_comedor_kwh',
    'energia_salones_kwh',
    'energia_laboratorios_kwh',
    'energia_auditorios_kwh',
    'energia_oficinas_kwh'
]
df[energia_cols] = df[energia_cols] * 1000

# Agua: viene dividida entre 100 → pasar a litros reales
df['agua_litros'] = df['agua_litros'] * 100

# CO2: viene en toneladas → pasar a kg reales
df['co2_kg'] = df['co2_kg'] * 1000

print("✅ Escalas corregidas:")
print("   ⚡ Energía → kWh reales")
print("   💧 Agua → litros reales")
print("   🌱 CO2 → kg reales")


# Mapeo de nombres de columnas de consumo por sector a nombres legibles
sectores_map = {
    'energia_comedor_kwh': 'Comedores',
    'energia_salones_kwh': 'Salones',
    'energia_laboratorios_kwh': 'Laboratorios',
    'energia_auditorios_kwh': 'Auditorios',
    'energia_oficinas_kwh': 'Oficinas'
}

# Columnas que se mantienen como variables de identificación en la
# transformación de formato wide a long
id_vars = [
    'timestamp', 'sede_id', 'temperatura_exterior_c', 'ocupacion_pct',
    'hora', 'dia_semana', 'es_festivo', 'es_semana_parciales',
    'es_semana_finales', 'co2_kg', 'agua_litros'
]

# Transformar de formato wide a long: cada sector ahora es una fila
# con su consumo correspondiente. Esto permite entrenar un modelo que
# generalice a todos los sectores.
df_global = pd.melt(
    df,
    id_vars=id_vars,
    value_vars=list(sectores_map.keys()),
    var_name='sector_original',
    value_name='consumo_kwh'
)

df_global['sector'] = df_global['sector_original'].map(sectores_map)
df_global.drop(columns=['sector_original'], inplace=True)

# Codificar nombres de sectores a números (0-4) para usar como features
# en el modelo XGBoost
le_sector = LabelEncoder()
df_global['sector_encoded'] = le_sector.fit_transform(df_global['sector'])

# ================================================================
# LIMPIEZA AVANZADA DE DATOS
# ================================================================
# Esta sección elimina outliers y valores anómalos que podrían afectar
# negativamente al entrenamiento del modelo. Utiliza percentiles para
# definir límites realistas en cada variable.

print("=" * 70)
print("🧹 LIMPIEZA AVANZADA DE DATOS")
print("=" * 70)

# Consumo: Eliminar picos extremos (mantener hasta P99)
# Los valores por encima del percentil 99 generalmente son anomalías
q_consumo_99 = df_global['consumo_kwh'].quantile(0.99)
df_global['consumo_kwh'] = df_global['consumo_kwh'].clip(upper=q_consumo_99)
print(f"✅ Consumo clippeado al P99: {q_consumo_99:.2f} kWh")

# CO2: Asegurar no negativos y eliminar extremos
df_global['co2_kg'] = df_global['co2_kg'].clip(lower=0)
co2_q99 = df_global['co2_kg'].quantile(0.99)
df_global['co2_kg'] = df_global['co2_kg'].clip(upper=co2_q99)
print(f"✅ CO2 clippeado: [0, {co2_q99:.2f}] kg")

# AGUA: Limpieza más agresiva (P98 vs P99) porque tiene distribución
# muy asimétrica con muchos outliers extremos
print(f"\n💧 AGUA - Antes:")
print(f"   Min: {df_global['agua_litros'].min():.2f}")
print(f"   Max: {df_global['agua_litros'].max():.2f}")
print(f"   Media: {df_global['agua_litros'].mean():.2f}")
print(f"   Std: {df_global['agua_litros'].std():.2f}")

# Eliminar outliers extremos (percentil 98 en lugar de 99)
agua_q98 = df_global['agua_litros'].quantile(0.98)
agua_q01 = df_global['agua_litros'].quantile(0.01)
df_global['agua_litros'] = df_global['agua_litros'].clip(lower=agua_q01, upper=agua_q98)

print(f"\n💧 AGUA - Después:")
print(f"   Min: {df_global['agua_litros'].min():.2f}")
print(f"   Max: {df_global['agua_litros'].max():.2f}")
print(f"   Clipped a: [{agua_q01:.2f}, {agua_q98:.2f}]")
print(f"   Nueva Std: {df_global['agua_litros'].std():.2f}")

# ================================================================
# ANÁLISIS DE CORRELACIÓN
# ================================================================
# Investigar relaciones entre consumo de agua y otros factores
# (sectores, horas, etc.) para informar el feature engineering

print("\n" + "=" * 70)
print("🔬 ANÁLISIS DE CORRELACIÓN CON AGUA")
print("=" * 70)

# Calcular consumo promedio de agua por sector
# Algunos sectores (ej: comedores) consumen más agua que otros
agua_por_sector = df_global.groupby('sector')['agua_litros'].agg(['mean', 'std'])
print("\n💧 Consumo de agua por sector:")
print(agua_por_sector)

# Identificar horas pico de consumo de agua
# Importante para predicciones más precisas según la hora del día
agua_por_hora = df_global.groupby('hora')['agua_litros'].mean()
horas_pico_agua = agua_por_hora.nlargest(5).index.tolist()
print(f"\n⏰ Horas con mayor consumo de agua: {horas_pico_agua}")

# ================================================================
# FEATURE ENGINEERING ESPECIALIZADO
# ================================================================
# Crear features sintéticas que capturen patrones complejos y relaciones
# entre variables. El feature engineering es crítico para mejorar la
# precisión del modelo, especialmente para el agua que tiene patrones
# no-lineales complejos.

print("\n" + "=" * 70)
print("🔧 FEATURE ENGINEERING ESPECIALIZADO")
print("=" * 70)

df_global['timestamp'] = pd.to_datetime(df_global['timestamp'])

# FEATURES CÍCLICAS: Convertir variables circulares (hora, día) a
# representación seno/coseno para capturar naturaleza cíclica
# Ej: la hora 23 es cercana a la hora 0, pero numéricamente 23 ≠ 0
df_global['hora_sin'] = np.sin(2 * np.pi * df_global['hora'] / 24)
df_global['hora_cos'] = np.cos(2 * np.pi * df_global['hora'] / 24)
df_global['dia_sin'] = np.sin(2 * np.pi * df_global['dia_semana'] / 7)
df_global['dia_cos'] = np.cos(2 * np.pi * df_global['dia_semana'] / 7)

# INTERACCIONES: Capturar efectos multiplicativos entre variables
df_global['temp_x_ocupacion'] = df_global['temperatura_exterior_c'] * df_global['ocupacion_pct']
df_global['consumo_x_ocupacion'] = df_global['consumo_kwh'] * df_global['ocupacion_pct']
df_global['consumo_cuadrado'] = df_global['consumo_kwh'] ** 2

# INDICADORES TEMPORALES: Flags para períodos especiales del día/semana
# que tienen patrones distintos de consumo
df_global['es_hora_pico'] = ((df_global['hora'] >= 8) & (df_global['hora'] <= 12) |
                              (df_global['hora'] >= 14) & (df_global['hora'] <= 18)).astype(int)
df_global['es_fin_semana'] = (df_global['dia_semana'] >= 5).astype(int)
df_global['es_noche'] = ((df_global['hora'] >= 22) | (df_global['hora'] <= 6)).astype(int)

# FEATURES ESPECÍFICAS PARA AGUA: El agua tiene patrones muy diferentes
# según el sector. Comedores consumen más agua en horas de comida.
df_global['es_hora_comida'] = df_global['hora'].isin([7, 8, 12, 13, 18, 19]).astype(int)
df_global['temp_alta'] = (df_global['temperatura_exterior_c'] > 25).astype(int)

# Interacción sector-temperatura: Comedores pueden ser más sensibles
# a temperatura que otros sectores
df_global['sector_comedor'] = (df_global['sector'] == 'Comedores').astype(int)
df_global['comedor_x_temp'] = df_global['sector_comedor'] * df_global['temperatura_exterior_c']
df_global['comedor_x_ocupacion'] = df_global['sector_comedor'] * df_global['ocupacion_pct']

# Laboratorios: Pueden tener consumo de agua para equipos
df_global['sector_laboratorio'] = (df_global['sector'] == 'Laboratorios').astype(int)
df_global['lab_x_ocupacion'] = df_global['sector_laboratorio'] * df_global['ocupacion_pct']

# ESTADÍSTICAS HISTÓRICAS: Promedios por grupo para capturar tendencias
# de largo plazo que el modelo puede usar como contexto
agua_sector_promedio = df_global.groupby('sector')['agua_litros'].mean().to_dict()
agua_sector_std = df_global.groupby('sector')['agua_litros'].std().to_dict()
df_global['agua_sector_promedio'] = df_global['sector'].map(agua_sector_promedio)
df_global['agua_sector_std'] = df_global['sector'].map(agua_sector_std)

# Promedio de agua por sede (cada edificio/campus puede tener características diferentes)
agua_sede_promedio = df_global.groupby('sede_id')['agua_litros'].mean().to_dict()
df_global['agua_sede_promedio'] = df_global['sede_id'].map(agua_sede_promedio)

# Emisiones promedio por sector (para usar como contexto en modelo de CO2)
emisiones_sector = df_global.groupby('sector')['co2_kg'].mean().to_dict()
df_global['emisiones_sector_promedio'] = df_global['sector'].map(emisiones_sector)

print("✅ Features creadas:")
print("   - 7 features cíclicas y temporales")
print("   - 6 features específicas para agua")
print("   - 5 features de estadísticas históricas")

# ================================================================
# TRANSFORMACIÓN LOGARÍTMICA PARA AGUA
# ================================================================
# El agua tiene una distribución altamente asimétrica (skewed). 
# Aplicar log(x) normaliza la distribución y mejora significativamente 
# la precisión del modelo (reduce RMSE y aumenta R²).

print("\n" + "=" * 70)
print("📐 TRANSFORMACIÓN DE AGUA")
print("=" * 70)

# Agua tiene distribución muy sesgada → aplicar log
# log1p = log(1 + x) para evitar problemas con log(0)
df_global['agua_litros_log'] = np.log1p(df_global['agua_litros'])

print(f"Agua original - Skewness: {df_global['agua_litros'].skew():.2f}")
print(f"Agua log      - Skewness: {df_global['agua_litros_log'].skew():.2f}")
print("✅ Transformación logarítmica aplicada")

# ================================================================
# PREPARACIÓN DE DATOS
# ================================================================
# División temporal: 80% entrenamiento, 20% prueba
# Se usa división temporal (no aleatoria) para respetar la naturaleza
# temporal de los datos energéticos

df_global = df_global.sort_values('timestamp').reset_index(drop=True)
split_idx = int(len(df_global) * 0.8)

print(f"\n📊 División temporal:")
print(f"   Entrenamiento: {split_idx:,} registros")
print(f"   Prueba: {len(df_global) - split_idx:,} registros")

# ================================================================
# ETAPA 1: MODELO PARA CONSUMO
# ================================================================
# Predictor base del consumo energético usando características tempotrales,
# de ocupación y ambientales. Las predicciones de esta etapa se usan como
# entrada en las etapas 2 y 3 (stacked generalization).

print("\n" + "=" * 70)
print("🔹 ETAPA 1: MODELO PARA CONSUMO ENERGÉTICO")
print("=" * 70)

# Seleccionar features disponibles en el momento de la predicción
features_stage1 = [
    'sede_id', 'sector_encoded', 'hora', 'hora_sin', 'hora_cos',
    'dia_semana', 'dia_sin', 'dia_cos', 'temperatura_exterior_c',
    'ocupacion_pct', 'temp_x_ocupacion', 'es_festivo',
    'es_semana_parciales', 'es_semana_finales', 'es_hora_pico',
    'es_fin_semana', 'es_noche', 'es_hora_comida'
]

X1_train = df_global[features_stage1][:split_idx]
X1_test = df_global[features_stage1][split_idx:]
y1_train = df_global['consumo_kwh'][:split_idx].values
y1_test = df_global['consumo_kwh'][split_idx:].values

# Configuración de XGBoost optimizada para consumo energético
# Hiperparámetros ajustados mediante experimentos previos
xgb_consumo = XGBRegressor(
    n_estimators=500,          # Número de árboles
    learning_rate=0.05,        # Tasa de aprendizaje (evita overfitting)
    max_depth=8,               # Profundidad máxima de árboles
    min_child_weight=3,        # Mínimo peso para nodo hoja
    subsample=0.8,             # % de datos por árbol (regularización)
    colsample_bytree=0.8,      # % de features por árbol
    gamma=0.1,                 # Regularización L1 en splits
    reg_alpha=0.1,             # Penalidad L1
    reg_lambda=1.0,            # Penalidad L2
    random_state=42,           # Reproducibilidad
    tree_method='hist',        # Método eficiente de construcción
    n_jobs=-1                  # Usar todos los cores disponibles
)

print("⏳ Entrenando modelo de consumo...")
xgb_consumo.fit(X1_train, y1_train)

pred_consumo_train = xgb_consumo.predict(X1_train)
pred_consumo_test = xgb_consumo.predict(X1_test)

# Guardar predicciones en dataframe para usarlas en etapa 2
df_global.loc[:split_idx-1, 'pred_consumo_kwh'] = pred_consumo_train
df_global.loc[split_idx:, 'pred_consumo_kwh'] = pred_consumo_test

print("✅ Modelo de consumo completado")

# Calcular métricas de evaluación
mae_consumo = mean_absolute_error(y1_test, pred_consumo_test)
rmse_consumo = np.sqrt(mean_squared_error(y1_test, pred_consumo_test))
r2_consumo = r2_score(y1_test, pred_consumo_test)
mape_consumo = np.mean(np.abs((y1_test - pred_consumo_test) / (y1_test + 1e-8))) * 100

print(f"   MAE:  {mae_consumo:.4f} kWh")
print(f"   RMSE: {rmse_consumo:.4f} kWh")
print(f"   R²:   {r2_consumo:.4f}")
print(f"   MAPE: {mape_consumo:.2f}%")

# ================================================================
# ETAPA 2: MODELO MEJORADO PARA AGUA
# ================================================================
# Modelo especializado para agua que usa:
# 1. Predicciones de consumo (de etapa 1)
# 2. Features específicas por sector (comedores vs laboratorios)
# 3. Transformación logarítmica (normaliza distribución sesgada)
# 4. Estadísticas históricas por sector/sede
#
# Esta es la "etapa crítica" - agua es la variable más difícil de predecir

print("\n" + "=" * 70)
print("💧 ETAPA 2: MODELO ESPECIALIZADO PARA AGUA")
print("=" * 70)

# Agregar predicciones de consumo como features (stacked generalization)
features_stage2 = features_stage1 + [
    'pred_consumo_kwh',
    'consumo_x_ocupacion',
    'temp_alta',
    'sector_comedor',
    'comedor_x_temp',
    'comedor_x_ocupacion',
    'sector_laboratorio',
    'lab_x_ocupacion',
    'agua_sector_promedio',
    'agua_sector_std',
    'agua_sede_promedio'
]

X2_train = df_global[features_stage2][:split_idx]
X2_test = df_global[features_stage2][split_idx:]

# Entrenar con AGUA transformada a escala logarítmica
# La predicción será en escala log, se convertirá de vuelta con expm1
y2_train_log = df_global['agua_litros_log'][:split_idx].values
y2_test_log = df_global['agua_litros_log'][split_idx:].values
y2_test_original = df_global['agua_litros'][split_idx:].values

xgb_agua = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=10,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.15,
    reg_alpha=0.2,
    reg_lambda=1.5,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

print("⏳ Entrenando modelo de agua (escala log)...")
xgb_agua.fit(X2_train, y2_train_log)

# Predecir en escala log y convertir de vuelta
# expm1(x) es la inversa de log1p(x): expm1(log1p(y)) ≈ y
pred_agua_log = xgb_agua.predict(X2_test)
pred_agua = np.expm1(pred_agua_log)

print("✅ Modelo de agua completado")

# Calcular métricas
mae_agua = mean_absolute_error(y2_test_original, pred_agua)
rmse_agua = np.sqrt(mean_squared_error(y2_test_original, pred_agua))
r2_agua = r2_score(y2_test_original, pred_agua)
mape_agua = np.mean(np.abs((y2_test_original - pred_agua) / (y2_test_original + 1))) * 100

print(f"\n💧 AGUA (MODELO MEJORADO)")
print(f"   MAE:      {mae_agua:>10.2f} litros 🚀")
print(f"   RMSE:     {rmse_agua:>10.2f} litros")
print(f"   R²:       {r2_agua:>10.4f} 🚀")
print(f"   MAPE:     {mape_agua:>10.2f}%")
print(f"   Media:    {y2_test_original.mean():>10.2f} litros")
print(f"   Error %:  {(mae_agua/y2_test_original.mean()*100):>10.2f}%")

# ================================================================
# ETAPA 3: MODELO PARA CO2
# ================================================================
# Predicción de emisiones de CO2 basado en:
# 1. Predicciones de consumo (etapa 1)
# 2. Predicciones de agua (etapa 2)
# 3. Features derivadas (consumo²)
# 4. Contexto histórico por sector
#
# El CO2 está fuertemente correlacionado con energía consumida

print("\n" + "=" * 70)
print("🌱 ETAPA 3: MODELO PARA CO2")
print("=" * 70)

df_global.loc[split_idx:, 'pred_agua_litros'] = pred_agua

# Features = todas las anteriores + predicciones de etapas 1 y 2
features_stage3 = features_stage2 + [
    'pred_agua_litros',
    'consumo_cuadrado',
    'emisiones_sector_promedio'
]

X3_train = df_global[features_stage3][:split_idx]
X3_test = df_global[features_stage3][split_idx:]
y3_train = df_global['co2_kg'][:split_idx].values
y3_test = df_global['co2_kg'][split_idx:].values

xgb_co2 = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=10,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.15,
    reg_alpha=0.2,
    reg_lambda=1.5,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

print("⏳ Entrenando modelo de CO2...")
xgb_co2.fit(X3_train, y3_train)
pred_co2 = xgb_co2.predict(X3_test)

print("✅ Modelo de CO2 completado")

# Calcular métricas
mae_co2 = mean_absolute_error(y3_test, pred_co2)
rmse_co2 = np.sqrt(mean_squared_error(y3_test, pred_co2))
r2_co2 = r2_score(y3_test, pred_co2)
mask = y3_test > 0.1
mape_co2 = np.mean(np.abs((y3_test[mask] - pred_co2[mask]) / y3_test[mask])) * 100

print(f"\n🌱 CO2 (MODELO FINAL)")
print(f"   MAE:      {mae_co2:>10.4f} kg")
print(f"   RMSE:     {rmse_co2:>10.4f} kg")
print(f"   R²:       {r2_co2:>10.4f}")
print(f"   MAPE:     {mape_co2:>10.2f}%")
print(f"   Media:    {y3_test.mean():>10.2f} kg")
print(f"   Error %:  {(mae_co2/y3_test.mean()*100):>10.2f}%")

# ================================================================
# RESUMEN FINAL
# ================================================================

print("\n" + "=" * 70)
print("📊 RESUMEN FINAL - TODAS LAS VARIABLES")
print("=" * 70)

print(f"\n⚡ CONSUMO ENERGÉTICO")
print(f"   MAE:  {mae_consumo:.4f} kWh")
print(f"   RMSE: {rmse_consumo:.4f} kWh")
print(f"   R²:   {r2_consumo:.4f}")
print(f"   MAPE: {mape_consumo:.2f}%")

print(f"\n💧 AGUA")
print(f"   MAE:  {mae_agua:.2f} litros")
print(f"   RMSE: {rmse_agua:.2f} litros")
print(f"   R²:   {r2_agua:.4f} 🚀")
print(f"   MAPE: {mape_agua:.2f}%")

print(f"\n🌱 CO2")
print(f"   MAE:  {mae_co2:.4f} kg")
print(f"   RMSE: {rmse_co2:.4f} kg")
print(f"   R²:   {r2_co2:.4f}")
print(f"   MAPE: {mape_co2:.2f}%")

# ================================================================
# IMPORTANCIA DE FEATURES PARA AGUA
# ================================================================

print("\n" + "=" * 70)
print("📈 IMPORTANCIA DE FEATURES PARA AGUA")
print("=" * 70)

importances_agua = xgb_agua.feature_importances_
importance_df_agua = pd.DataFrame({
    'Feature': features_stage2,
    'Importancia': importances_agua
}).sort_values('Importancia', ascending=False)

print(importance_df_agua.head(15).to_string(index=False))

# ================================================================
# IMPORTANCIA DE FEATURES PARA CO2
# ================================================================

print("\n" + "=" * 70)
print("📈 IMPORTANCIA DE FEATURES PARA CO2")
print("=" * 70)

importances_co2 = xgb_co2.feature_importances_
importance_df_co2 = pd.DataFrame({
    'Feature': features_stage3,
    'Importancia': importances_co2
}).sort_values('Importancia', ascending=False)

print(importance_df_co2.head(15).to_string(index=False))

# ================================================================
# ANÁLISIS DE ERRORES AGUA
# ================================================================

print("\n" + "=" * 70)
print("🔍 ANÁLISIS DE ERRORES - AGUA")
print("=" * 70)

results_agua = df_global.iloc[split_idx:].copy()
results_agua['pred_agua_litros'] = pred_agua
results_agua['error_agua'] = np.abs(results_agua['agua_litros'] - pred_agua)

print("\n🔴 TOP 10 MAYORES ERRORES EN AGUA:")
top_errors_agua = results_agua.nlargest(10, 'error_agua')[
    ['timestamp', 'sector', 'sede_id', 'ocupacion_pct', 'agua_litros', 'pred_agua_litros', 'error_agua']
]
print(top_errors_agua.to_string(index=False))

print("\n📊 ERROR PROMEDIO DE AGUA POR SECTOR:")
error_by_sector_agua = results_agua.groupby('sector')['error_agua'].agg(['mean', 'std', 'max'])
print(error_by_sector_agua)

print("\n🏢 ERROR PROMEDIO DE AGUA POR SEDE:")
error_by_sede_agua = results_agua.groupby('sede_id')['error_agua'].agg(['mean', 'std'])
print(error_by_sede_agua)

# ================================================================
# ANÁLISIS DE ERRORES CO2
# ================================================================

print("\n" + "=" * 70)
print("🔍 ANÁLISIS DE ERRORES - CO2")
print("=" * 70)

results_co2 = df_global.iloc[split_idx:].copy()
results_co2['pred_co2_kg'] = pred_co2
results_co2['error_co2'] = np.abs(results_co2['co2_kg'] - pred_co2)

print("\n🔴 TOP 10 MAYORES ERRORES EN CO2:")
top_errors_co2 = results_co2.nlargest(10, 'error_co2')[
    ['timestamp', 'sector', 'sede_id', 'consumo_kwh', 'co2_kg', 'pred_co2_kg', 'error_co2']
]
print(top_errors_co2.to_string(index=False))

print("\n📊 ERROR PROMEDIO DE CO2 POR SECTOR:")
error_by_sector_co2 = results_co2.groupby('sector')['error_co2'].agg(['mean', 'std', 'max'])
print(error_by_sector_co2)

# ================================================================
# GUARDAR MODELOS
# ================================================================
# Se guardan todos los modelos entrenados y su configuración para
# usar en predicciones posteriores sin necesidad de reentrenamiento

print("\n" + "=" * 70)
print("💾 GUARDANDO MODELOS OPTIMIZADOS")
print("=" * 70)

joblib.dump(xgb_consumo, 'modelo_consumo.pkl')
joblib.dump(xgb_agua, 'modelo_agua_mejorado.pkl')
joblib.dump(xgb_co2, 'modelo_co2.pkl')
joblib.dump(le_sector, 'label_encoder_sector.pkl')
joblib.dump({
    'features_stage1': features_stage1,
    'features_stage2': features_stage2,
    'features_stage3': features_stage3,
    'emisiones_sector': emisiones_sector,
    'agua_sector_promedio': agua_sector_promedio,
    'agua_sector_std': agua_sector_std,
    'agua_sede_promedio': agua_sede_promedio
}, 'config_features.pkl')

print("✅ Modelo Consumo: modelo_consumo.pkl")
print("✅ Modelo Agua: modelo_agua_mejorado.pkl")
print("✅ Modelo CO2: modelo_co2.pkl")
print("✅ Configuración: config_features.pkl")

# ================================================================
# FUNCIÓN DE PREDICCIÓN FINAL
# ================================================================

def predecir_completo(sede_id, sector, hora, dia_semana, temperatura, ocupacion,
                       es_festivo=0, es_parciales=0, es_finales=0):
    """
    Predicción completa en 3 etapas: Consumo → Agua → CO2
    
    Esta función hace predicciones usando los 3 modelos entrenados en cascada.
    Las predicciones de una etapa se usan como entrada en la siguiente.
    
    Parámetros:
    -----------
    sede_id : int
        Identificador de la sede/campus (1-4 típicamente)
    sector : str
        Nombre del sector: 'Comedores', 'Laboratorios', 'Salones', 
        'Auditorios', 'Oficinas'
    hora : int
        Hora del día (0-23)
    dia_semana : int
        Día de la semana (0=lunes, 6=domingo)
    temperatura : float
        Temperatura exterior en °C
    ocupacion : float
        Porcentaje de ocupación del edificio (0-100)
    es_festivo : int, optional
        1 si es día festivo, 0 por defecto
    es_parciales : int, optional
        1 si hay exámenes parciales, 0 por defecto
    es_finales : int, optional
        1 si hay exámenes finales, 0 por defecto
    
    Retorna:
    --------
    dict : Predicciones con claves:
        - 'consumo_kwh': Consumo energético predicho (kWh)
        - 'agua_litros': Consumo de agua predicho (litros)
        - 'co2_kg': Emisiones CO2 predichas (kg)
    """
    # Cargar modelos y configuración guardados en archivos
    model_consumo = joblib.load('modelo_consumo.pkl')
    model_agua = joblib.load('modelo_agua_mejorado.pkl')
    model_co2 = joblib.load('modelo_co2.pkl')
    encoder = joblib.load('label_encoder_sector.pkl')
    config = joblib.load('config_features.pkl')

    # Codificar nombre de sector a número (0-4)
    sector_encoded = encoder.transform([sector])[0]

    # Calcular features cíclicas
    hora_sin = np.sin(2 * np.pi * hora / 24)
    hora_cos = np.cos(2 * np.pi * hora / 24)
    dia_sin = np.sin(2 * np.pi * dia_semana / 7)
    dia_cos = np.cos(2 * np.pi * dia_semana / 7)
    temp_x_ocupacion = temperatura * ocupacion
    es_hora_pico = int((8 <= hora <= 12) or (14 <= hora <= 18))
    es_fin_semana = int(dia_semana >= 5)
    es_noche = int((hora >= 22) or (hora <= 6))
    es_hora_comida = int(hora in [7, 8, 12, 13, 18, 19])

    # ETAPA 1: Predecir consumo energético
    X1 = pd.DataFrame([[
        sede_id, sector_encoded, hora, hora_sin, hora_cos,
        dia_semana, dia_sin, dia_cos, temperatura, ocupacion,
        temp_x_ocupacion, es_festivo, es_parciales, es_finales,
        es_hora_pico, es_fin_semana, es_noche, es_hora_comida
    ]], columns=config['features_stage1'])

    pred_consumo = model_consumo.predict(X1)[0]

    # ETAPA 2: Predecir agua usando predicción de consumo
    consumo_x_ocupacion = pred_consumo * ocupacion
    temp_alta = int(temperatura > 25)
    sector_comedor = int(sector == 'Comedores')
    comedor_x_temp = sector_comedor * temperatura
    comedor_x_ocupacion = sector_comedor * ocupacion
    sector_laboratorio = int(sector == 'Laboratorios')
    lab_x_ocupacion = sector_laboratorio * ocupacion
    agua_sector_prom = config['agua_sector_promedio'][sector]
    agua_sector_std_val = config['agua_sector_std'][sector]
    agua_sede_prom = config['agua_sede_promedio'][sede_id]

    X2 = pd.DataFrame([[
        sede_id, sector_encoded, hora, hora_sin, hora_cos,
        dia_semana, dia_sin, dia_cos, temperatura, ocupacion,
        temp_x_ocupacion, es_festivo, es_parciales, es_finales,
        es_hora_pico, es_fin_semana, es_noche, es_hora_comida,
        pred_consumo, consumo_x_ocupacion, temp_alta,
        sector_comedor, comedor_x_temp, comedor_x_ocupacion,
        sector_laboratorio, lab_x_ocupacion, agua_sector_prom,
        agua_sector_std_val, agua_sede_prom
    ]], columns=config['features_stage2'])

    pred_agua_log = model_agua.predict(X2)[0]
    pred_agua = np.expm1(pred_agua_log)

    # ETAPA 3: Predecir CO2 usando predicciones anteriores
    consumo_cuadrado = pred_consumo ** 2
    emisiones_promedio = config['emisiones_sector'][sector]

    X3 = pd.DataFrame([[
        sede_id, sector_encoded, hora, hora_sin, hora_cos,
        dia_semana, dia_sin, dia_cos, temperatura, ocupacion,
        temp_x_ocupacion, es_festivo, es_parciales, es_finales,
        es_hora_pico, es_fin_semana, es_noche, es_hora_comida,
        pred_consumo, consumo_x_ocupacion, temp_alta,
        sector_comedor, comedor_x_temp, comedor_x_ocupacion,
        sector_laboratorio, lab_x_ocupacion, agua_sector_prom,
        agua_sector_std_val, agua_sede_prom, pred_agua,
        consumo_cuadrado, emisiones_promedio
    ]], columns=config['features_stage3'])

    pred_co2 = model_co2.predict(X3)[0]

    # Retornar predicciones redondeadas a 2 decimales
    return {
        'consumo_kwh': round(float(pred_consumo), 2),
        'agua_litros': round(float(pred_agua), 2),
        'co2_kg': round(float(pred_co2), 2)
    }

# ================================================================
# EJEMPLOS DE PREDICCIÓN
# ================================================================
# Demostración de cómo usar la función predecir_completo() con
# diferentes escenarios realistas

print("\n" + "=" * 70)
print("🧪 EJEMPLOS DE PREDICCIÓN")
print("=" * 70)

# Definir 4 escenarios de prueba con diferentes combinaciones de parámetros
ejemplos = [
    # Comedor a la hora de almuerzo, ocupación alta
    {'sede': 1, 'sector': 'Comedores', 'hora': 12, 'dia': 2, 'temp': 25, 'ocup': 75},
    # Laboratorio a media mañana, ocupación muy alta
    {'sede': 3, 'sector': 'Laboratorios', 'hora': 10, 'dia': 3, 'temp': 22, 'ocup': 85},
    # Salón por la tarde, temperatura alta
    {'sede': 2, 'sector': 'Salones', 'hora': 15, 'dia': 1, 'temp': 28, 'ocup': 60},
    # Oficinas por la mañana, ocupación máxima
    {'sede': 4, 'sector': 'Oficinas', 'hora': 9, 'dia': 4, 'temp': 20, 'ocup': 90},
]

# Ejecutar predicciones para cada escenario
for ej in ejemplos:
    pred = predecir_completo(
        sede_id=ej['sede'], sector=ej['sector'], hora=ej['hora'],
        dia_semana=ej['dia'], temperatura=ej['temp'], ocupacion=ej['ocup']
    )
    dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    print(f"\n📍 Sede {ej['sede']}, {ej['sector']}, {dias[ej['dia']]} {ej['hora']}:00, {ej['temp']}°C, {ej['ocup']}% ocupación")
    print(f"   ⚡ Consumo: {pred['consumo_kwh']} kWh")
    print(f"   💧 Agua: {pred['agua_litros']} litros")
    print(f"   🌱 CO2: {pred['co2_kg']} kg")

print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETO CON 3 ETAPAS FINALIZADO")
print("=" * 70)