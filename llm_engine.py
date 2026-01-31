"""
Motor de predicción ML para análisis energético universitario.
Usa modelos XGBoost entrenados para predecir consumo, agua y CO2.
"""

import joblib
import numpy as np
import pandas as pd

# =====================================================
# CARGAR MODELOS ML
# =====================================================
modelo_consumo = modelo_agua = modelo_co2 = config_features = None
try:
    print("⏳ Cargando modelos ML...")
    modelo_consumo = joblib.load("models/modelo_consumo.pkl")
    modelo_agua = joblib.load("models/modelo_agua_mejorado.pkl")
    modelo_co2 = joblib.load("models/modelo_co2.pkl")
    config_features = joblib.load("models/config_features.pkl")
    print("✅ Modelos ML cargados")
except Exception as e:
    print(f"⚠️ Modelos ML no disponibles: {e}")

# =====================================================
# CARGAR DATOS CSV
# =====================================================
df_energia = None
try:
    df_energia = pd.read_csv("data/dataset_energia_limpio_sectores.csv")
    print(f"✅ CSV cargado: {len(df_energia)} registros")
except Exception as e:
    print(f"❌ Error cargando CSV: {e}")

# =====================================================
# CONSTANTES
# =====================================================
SEDES = {
    1: "Chiquinquirá",
    2: "Tunja", 
    3: "Duitama",
    4: "Sogamoso"
}

SECTORES = {
    'Comedores': 'comedor',
    'Salones': 'salon',
    'Laboratorios': 'laboratorio',
    'Auditorios': 'auditorio',
    'Oficinas': 'oficina'
}

# =====================================================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN
# =====================================================
def predecir_completo(sede_id, sector, hora, dia_semana, temperatura, ocupacion):
    """
    Predicción completa usando 3 modelos en cascada.
    
    Args:
        sede_id: ID de sede (1-4)
        sector: Nombre del sector (Comedores, Salones, etc.)
        hora: Hora del día (0-23)
        dia_semana: Día de la semana (0=lunes, 6=domingo)
        temperatura: Temperatura exterior en °C
        ocupacion: Porcentaje de ocupación (0-100)
    
    Returns:
        dict con energia_kwh, agua_litros, co2_kg
    """
    try:
        features_consumo = config_features.get("features_stage1", []) if config_features else []
        
        # Preparar features
        X_dict = {
            'ocupacion_pct': ocupacion,
            'temperatura_exterior_c': temperatura,
            'hora': hora,
            'dia_semana': dia_semana,
            'sede_id': sede_id,
        }
        
        # One-hot encoding de sectores
        for sect_name, sect_col in SECTORES.items():
            X_dict[f'sector_{sect_col}'] = 1 if sect_name == sector else 0
        
        # Stage 1: Consumo energético
        X_consumo = [X_dict.get(f, 0) for f in features_consumo]
        X_consumo = np.array([X_consumo])
        pred_consumo = modelo_consumo.predict(X_consumo)[0] if modelo_consumo else 0
        
        # Stage 2: Agua
        features_agua = config_features.get("features_stage2", []) if config_features else []
        X_agua = []
        for feat in features_agua:
            if feat == 'pred_consumo_kwh':
                X_agua.append(pred_consumo)
            else:
                X_agua.append(X_dict.get(feat, 0))
        X_agua = np.array([X_agua])
        pred_agua = modelo_agua.predict(X_agua)[0] if modelo_agua else 0
        
        # Stage 3: CO2
        features_co2 = config_features.get("features_stage3", []) if config_features else []
        X_co2 = []
        for feat in features_co2:
            if feat == 'pred_consumo_kwh':
                X_co2.append(pred_consumo)
            elif feat == 'pred_agua_litros':
                X_co2.append(pred_agua)
            else:
                X_co2.append(X_dict.get(feat, 0))
        X_co2 = np.array([X_co2])
        pred_co2 = modelo_co2.predict(X_co2)[0] if modelo_co2 else 0
        
        return {
            'energia_kwh': float(round(pred_consumo, 2)),
            'agua_litros': float(round(pred_agua, 2)),
            'co2_kg': float(round(pred_co2, 2)),
            'sede': SEDES.get(sede_id, f"Sede {sede_id}"),
            'sector': sector
        }
    
    except Exception as e:
        print(f"⚠️ Error predicción: {e}")
        return {
            'energia_kwh': 0,
            'agua_litros': 0,
            'co2_kg': 0,
            'sede': SEDES.get(sede_id, f"Sede {sede_id}"),
            'sector': sector
        }
