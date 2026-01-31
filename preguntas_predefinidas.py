import random
from datetime import datetime, timedelta
import re
from llm_engine import predecir_completo as predecir_ml, SEDES

# =====================================================
# Preguntas predefinidas
# =====================================================
PREGUNTAS_PREDEFINIDAS = [
    "¿Cuánta energía consumirá el laboratorio de la sede 3 mañana a las 5pm?",
    "Dime el consumo de los comedores de la sede 2 el martes a las 12 del mediodía.",
    "¿Cuál será el consumo de oficinas en la sede 1 hoy a las 10:00 am?",
    "¿Cuánta agua se gastará en los auditorios de la sede 4 a las 7pm?",
    "¿Qué consumo tendrán los salones de la sede 2 el viernes a las 3pm?"
]

# =====================================================
# Detectar sector
# =====================================================
def detectar_sector(texto):
    t = texto.lower()
    if "laboratorio" in t:
        return "Laboratorios"
    elif "comedor" in t:
        return "Comedores"
    elif "oficina" in t:
        return "Oficinas"
    elif "auditorio" in t:
        return "Auditorios"
    elif "salon" in t or "salón" in t:
        return "Salones"
    else:
        return "General"

# =====================================================
# Detectar sede
# =====================================================
def extraer_sede(texto):
    match = re.search(r'sede\s*(\d)', texto.lower())
    if match:
        return int(match.group(1))
    return 2  # default Tunja

# =====================================================
# Detectar fecha y hora
# =====================================================
def extraer_fecha_hora(texto):
    ahora = datetime.now()
    texto_lower = texto.lower()
    
    # Detectar fecha
    if "mañana" in texto_lower:
        fecha = ahora + timedelta(days=1)
    elif "hoy" in texto_lower:
        fecha = ahora
    elif "martes" in texto_lower:
        dias_hasta = (1 - ahora.weekday() + 7) % 7
        if dias_hasta == 0:
            dias_hasta = 7
        fecha = ahora + timedelta(days=dias_hasta)
    elif "viernes" in texto_lower:
        dias_hasta = (4 - ahora.weekday() + 7) % 7
        if dias_hasta == 0:
            dias_hasta = 7
        fecha = ahora + timedelta(days=dias_hasta)
    else:
        fecha = ahora
    
    # Detectar hora
    hora = 9  # default
    
    # Buscar patrones de hora
    if "5pm" in texto_lower or "5 pm" in texto_lower:
        hora = 17
    elif "12" in texto and ("mediodía" in texto_lower or "mediodia" in texto_lower):
        hora = 12
    elif "10:00 am" in texto_lower or "10 am" in texto_lower:
        hora = 10
    elif "7pm" in texto_lower or "7 pm" in texto_lower:
        hora = 19
    elif "3pm" in texto_lower or "3 pm" in texto_lower:
        hora = 15
    
    return fecha, hora

# =====================================================
# Función principal para responder pregunta
# =====================================================
def responder_pregunta(pregunta):
    """
    Procesa una pregunta y genera la predicción usando los modelos ML
    """
    sector = detectar_sector(pregunta)
    sede_id = extraer_sede(pregunta)
    fecha, hora = extraer_fecha_hora(pregunta)
    
    # Obtener nombre de la sede
    nombre_sede = SEDES.get(sede_id, f"Sede {sede_id}")
    
    # Obtener día de la semana (0=lunes, 6=domingo)
    dia_semana = fecha.weekday()
    
    # Temperatura y ocupación basadas en hora
    if 6 <= hora <= 18:
        temperatura = random.randint(20, 28)
        ocupacion = random.randint(60, 90)
    else:
        temperatura = random.randint(15, 22)
        ocupacion = random.randint(10, 40)
    
    # Llamar a la predicción ML real
    try:
        resultado = predecir_ml(
            sede_id=sede_id,
            sector=sector,
            hora=hora,
            dia_semana=dia_semana,
            temperatura=temperatura,
            ocupacion=ocupacion
        )
        
        # Convertir numpy float32 a float de Python
        kwh = float(resultado.get("energia_kwh", 0))
        agua = float(resultado.get("agua_litros", 0))
        co2 = float(resultado.get("co2_kg", 0))
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        # Valores de fallback
        kwh = 1500.0
        agua = 30000.0
        co2 = 450.0
    
    # Formatear fecha en español
    dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    dia_nombre = dias_es[fecha.weekday()]
    fecha_formateada = f"{fecha.strftime('%d/%m/%Y')} - {dia_nombre}"
    
    respuesta = {
        "pregunta": pregunta,
        "sector": sector,
        "sede_id": sede_id,
        "sede_nombre": nombre_sede,
        "fecha": fecha_formateada,
        "hora": hora,
        "temperatura": temperatura,
        "ocupacion": ocupacion,
        "prediccion": {
            "energia_kwh": round(kwh, 2),
            "agua_litros": round(agua, 2),
            "co2_kg": round(co2, 2)
        },
        "respuesta_texto": (
            f"⚡ **Predicción para {sector} en {nombre_sede}**\n\n"
            f"📅 **Fecha**: {fecha_formateada}\n"
            f"⏰ **Hora**: {hora}:00\n"
            f"💡 **Consumo de energía**: {kwh:,.2f} kWh\n"
            f"💧 **Consumo de agua**: {agua:,.2f} litros\n"
            f"🌱 **Emisiones de CO₂**: {co2:,.2f} kg\n\n"
            f"📊 **Condiciones**: Temperatura ~{temperatura}°C, Ocupación ~{ocupacion}%"
        )
    }
    
    return respuesta

# =====================================================
# Obtener preguntas con índices
# =====================================================
def obtener_preguntas():
    """Retorna las preguntas predefinidas con sus índices"""
    return [
        {"id": i, "pregunta": p} 
        for i, p in enumerate(PREGUNTAS_PREDEFINIDAS, 1)
    ]
