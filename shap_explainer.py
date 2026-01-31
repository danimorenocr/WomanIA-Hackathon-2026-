import joblib

# Cargar modelos
modelo_consumo = joblib.load("models/modelo_consumo.pkl")
modelo_agua = joblib.load("models/modelo_agua_mejorado.pkl")
modelo_co2 = joblib.load("models/modelo_co2.pkl")

config_features = joblib.load("models/config_features.pkl")

import shap
import numpy as np

def generar_shap_resumen(modelo, X_sample, feature_names):
    explainer = shap.Explainer(modelo)
    shap_values = explainer(X_sample)

    resumen = {}
    for i, name in enumerate(feature_names):
        resumen[name] = float(np.mean(shap_values.values[:, i]))

    return resumen