from llm_engine import explicar


def responder_pregunta(pregunta):
    """
    El usuario pregunta → Modelo predice → SHAP explica → Respuesta inteligente
    Sin dependencia de JSON.
    """
    return explicar(None, pregunta)
