import json
import pandas as pd
import numpy as np
from llm_engine import df_energia, SEDES, predecir_completo

def generar_consumo_total_por_sede():
    """
    Consumo Energético Total por Sede
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    datos = []
    for sede_id, sede_nombre in SEDES.items():
        df_sede = df_energia[df_energia['sede_id'] == sede_id]
        if len(df_sede) > 0:
            datos.append({
                "sede": sede_nombre,
                "consumo_total_kwh": round(df_sede['energia_total_kwh'].sum(), 2),
                "consumo_promedio_kwh": round(df_sede['energia_total_kwh'].mean(), 2)
            })
    
    return {
        "title": "Consumo Energético Total por Sede",
        "data": sorted(datos, key=lambda x: x['consumo_total_kwh'], reverse=True)
    }


def generar_tendencias_consumo_por_sede():
    """
    Tendencias de Consumo por Sede (últimos 30 días)
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    # Convertir timestamp a datetime
    df_energia['timestamp'] = pd.to_datetime(df_energia['timestamp'])
    
    # Últimos 30 días
    fecha_limite = df_energia['timestamp'].max() - pd.Timedelta(days=30)
    df_reciente = df_energia[df_energia['timestamp'] >= fecha_limite]
    
    # Agrupar por día y sede
    df_reciente['fecha'] = df_reciente['timestamp'].dt.date
    
    tendencias = {}
    for sede_id, sede_nombre in SEDES.items():
        df_sede = df_reciente[df_reciente['sede_id'] == sede_id]
        if len(df_sede) > 0:
            consumo_diario = df_sede.groupby('fecha')['energia_total_kwh'].sum().reset_index()
            tendencias[sede_nombre] = [
                {
                    "fecha": str(row['fecha']),
                    "consumo_kwh": round(row['energia_total_kwh'], 2)
                }
                for _, row in consumo_diario.iterrows()
            ]
    
    return {
        "title": "Tendencias de Consumo por Sede (Últimos 30 días)",
        "data": tendencias
    }


def generar_eficiencia_por_estudiante():
    """
    Eficiencia Energética por Estudiante
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    datos = []
    for sede_id, sede_nombre in SEDES.items():
        df_sede = df_energia[df_energia['sede_id'] == sede_id]
        if len(df_sede) > 0:
            consumo_total = df_sede['energia_total_kwh'].sum()
            estudiantes = df_sede['num_estudiantes'].iloc[0] if 'num_estudiantes' in df_sede.columns else 1000
            
            datos.append({
                "sede": sede_nombre,
                "consumo_por_estudiante_kwh": round(consumo_total / estudiantes, 2) if estudiantes > 0 else 0,
                "estudiantes": int(estudiantes)
            })
    
    return {
        "title": "Eficiencia Energética por Estudiante",
        "data": sorted(datos, key=lambda x: x['consumo_por_estudiante_kwh'], reverse=True)
    }


def generar_emisiones_co2_por_sede():
    """
    Emisiones de CO₂ por Sede
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    datos = []
    for sede_id, sede_nombre in SEDES.items():
        df_sede = df_energia[df_energia['sede_id'] == sede_id]
        if len(df_sede) > 0:
            co2_total_kg = df_sede['co2_kg'].sum()
            co2_total_ton = co2_total_kg / 1000
            
            datos.append({
                "sede": sede_nombre,
                "co2_total_toneladas": round(co2_total_ton, 2),
                "co2_promedio_kg_hora": round(df_sede['co2_kg'].mean(), 2)
            })
    
    return {
        "title": "Emisiones de CO₂ por Sede",
        "data": sorted(datos, key=lambda x: x['co2_total_toneladas'], reverse=True)
    }


def generar_consumo_agua_por_sede():
    """
    Consumo de Agua por Sede
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    datos = []
    for sede_id, sede_nombre in SEDES.items():
        df_sede = df_energia[df_energia['sede_id'] == sede_id]
        if len(df_sede) > 0:
            agua_total = df_sede['agua_litros'].sum()
            agua_total_m3 = agua_total / 1000
            
            datos.append({
                "sede": sede_nombre,
                "agua_total_m3": round(agua_total_m3, 2),
                "agua_promedio_litros_hora": round(df_sede['agua_litros'].mean(), 2)
            })
    
    return {
        "title": "Consumo de Agua por Sede",
        "data": sorted(datos, key=lambda x: x['agua_total_m3'], reverse=True)
    }


def generar_temperatura_vs_consumo():
    """
    Relación Temperatura vs Consumo Energético
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    # Agrupar por rangos de temperatura
    df_energia['rango_temp'] = pd.cut(
        df_energia['temperatura_exterior_c'], 
        bins=[0, 10, 15, 20, 25, 30, 100],
        labels=['0-10°C', '10-15°C', '15-20°C', '20-25°C', '25-30°C', '>30°C']
    )
    
    consumo_por_temp = df_energia.groupby('rango_temp')['energia_total_kwh'].mean().reset_index()
    
    datos = [
        {
            "rango_temperatura": str(row['rango_temp']),
            "consumo_promedio_kwh": round(row['energia_total_kwh'], 2)
        }
        for _, row in consumo_por_temp.iterrows()
    ]
    
    return {
        "title": "Relación Temperatura vs Consumo Energético",
        "data": datos
    }


def generar_consumo_por_sector():
    """
    Consumo por Sector (adicional)
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    sectores = {
        "Comedores": df_energia['energia_comedor_kwh'].sum(),
        "Salones": df_energia['energia_salones_kwh'].sum(),
        "Laboratorios": df_energia['energia_laboratorios_kwh'].sum(),
        "Auditorios": df_energia['energia_auditorios_kwh'].sum(),
        "Oficinas": df_energia['energia_oficinas_kwh'].sum()
    }
    
    datos = [
        {"sector": sector, "consumo_total_kwh": round(consumo, 2)}
        for sector, consumo in sorted(sectores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return {
        "title": "Consumo Energético por Sector",
        "data": datos
    }


def generar_distribucion_consumo_por_sector():
    """
    Distribución de Consumo por Sector (porcentajes)
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    sectores = {
        "Comedores": df_energia['energia_comedor_kwh'].sum(),
        "Salones": df_energia['energia_salones_kwh'].sum(),
        "Laboratorios": df_energia['energia_laboratorios_kwh'].sum(),
        "Auditorios": df_energia['energia_auditorios_kwh'].sum(),
        "Oficinas": df_energia['energia_oficinas_kwh'].sum()
    }
    
    total = sum(sectores.values())
    
    datos = [
        {
            "sector": sector,
            "consumo_kwh": round(consumo, 2),
            "porcentaje": round((consumo / total * 100), 2)
        }
        for sector, consumo in sorted(sectores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return {
        "title": "Distribución de Consumo por Sector",
        "data": datos
    }


def generar_tendencias_consumo_por_sector():
    """
    Tendencias de Consumo por Sector (últimos 30 días)
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    df_energia['timestamp'] = pd.to_datetime(df_energia['timestamp'])
    
    # Últimos 30 días
    fecha_limite = df_energia['timestamp'].max() - pd.Timedelta(days=30)
    df_reciente = df_energia[df_energia['timestamp'] >= fecha_limite].copy()
    
    df_reciente['fecha'] = df_reciente['timestamp'].dt.date
    
    sectores_data = {}
    sectores_cols = {
        "Comedores": "energia_comedor_kwh",
        "Salones": "energia_salones_kwh",
        "Laboratorios": "energia_laboratorios_kwh",
        "Auditorios": "energia_auditorios_kwh",
        "Oficinas": "energia_oficinas_kwh"
    }
    
    for sector_nombre, col in sectores_cols.items():
        consumo_diario = df_reciente.groupby('fecha')[col].sum().reset_index()
        sectores_data[sector_nombre] = [
            {
                "fecha": str(row['fecha']),
                "consumo_kwh": round(row[col], 2)
            }
            for _, row in consumo_diario.iterrows()
        ]
    
    return {
        "title": "Tendencias de Consumo por Sector (Últimos 30 días)",
        "data": sectores_data
    }


def generar_eficiencia_por_sector_y_sede():
    """
    Eficiencia Energética por Sector y Sede
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    datos = []
    sectores_cols = {
        "Comedores": "energia_comedor_kwh",
        "Salones": "energia_salones_kwh",
        "Laboratorios": "energia_laboratorios_kwh",
        "Auditorios": "energia_auditorios_kwh",
        "Oficinas": "energia_oficinas_kwh"
    }
    
    for sede_id, sede_nombre in SEDES.items():
        df_sede = df_energia[df_energia['sede_id'] == sede_id]
        if len(df_sede) > 0:
            for sector_nombre, col in sectores_cols.items():
                consumo_promedio = df_sede[col].mean()
                consumo_por_m2 = consumo_promedio / df_sede['area_m2'].iloc[0] if 'area_m2' in df_sede.columns and df_sede['area_m2'].iloc[0] > 0 else 0
                
                datos.append({
                    "sede": sede_nombre,
                    "sector": sector_nombre,
                    "consumo_promedio_kwh": round(consumo_promedio, 2),
                    "consumo_por_m2": round(consumo_por_m2, 4)
                })
    
    return {
        "title": "Eficiencia Energética por Sector y Sede",
        "data": datos
    }


def generar_correlacion_ocupacion_consumo():
    """
    Correlación Ocupación vs Consumo
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    # Agrupar por rangos de ocupación
    df_energia['rango_ocupacion'] = pd.cut(
        df_energia['ocupacion_pct'], 
        bins=[0, 20, 40, 60, 80, 100],
        labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    )
    
    consumo_por_ocupacion = df_energia.groupby('rango_ocupacion', observed=False).agg({
        'energia_total_kwh': 'mean',
        'agua_litros': 'mean',
        'co2_kg': 'mean'
    }).reset_index()
    
    datos = [
        {
            "rango_ocupacion": str(row['rango_ocupacion']),
            "consumo_promedio_kwh": round(row['energia_total_kwh'], 2),
            "agua_promedio_litros": round(row['agua_litros'], 2),
            "co2_promedio_kg": round(row['co2_kg'], 2)
        }
        for _, row in consumo_por_ocupacion.iterrows()
    ]
    
    return {
        "title": "Correlación Ocupación vs Consumo",
        "data": datos
    }


def generar_costos_operacionales_por_sector():
    """
    Costos Operacionales por Sector (asumiendo tarifa promedio)
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    # Tarifa promedio por kWh en Colombia (COP)
    tarifa_energia_cop = 650
    tarifa_agua_cop = 3.5  # por litro
    
    sectores_cols = {
        "Comedores": "energia_comedor_kwh",
        "Salones": "energia_salones_kwh",
        "Laboratorios": "energia_laboratorios_kwh",
        "Auditorios": "energia_auditorios_kwh",
        "Oficinas": "energia_oficinas_kwh"
    }
    
    datos = []
    for sector_nombre, col in sectores_cols.items():
        consumo_energia = df_energia[col].sum()
        costo_energia = consumo_energia * tarifa_energia_cop
        
        # Estimar agua proporcional al consumo energético
        proporcion = consumo_energia / df_energia['energia_total_kwh'].sum()
        consumo_agua = df_energia['agua_litros'].sum() * proporcion
        costo_agua = consumo_agua * tarifa_agua_cop
        
        costo_total = costo_energia + costo_agua
        
        datos.append({
            "sector": sector_nombre,
            "consumo_energia_kwh": round(consumo_energia, 2),
            "costo_energia_cop": round(costo_energia, 2),
            "consumo_agua_litros": round(consumo_agua, 2),
            "costo_agua_cop": round(costo_agua, 2),
            "costo_total_cop": round(costo_total, 2)
        })
    
    return {
        "title": "Costos Operacionales por Sector",
        "data": sorted(datos, key=lambda x: x['costo_total_cop'], reverse=True)
    }


def generar_impacto_ambiental_por_sector():
    """
    Impacto Ambiental por Sector (CO2 y huella hídrica)
    """
    if df_energia is None:
        return {"error": "Datos no disponibles"}
    
    sectores_cols = {
        "Comedores": "energia_comedor_kwh",
        "Salones": "energia_salones_kwh",
        "Laboratorios": "energia_laboratorios_kwh",
        "Auditorios": "energia_auditorios_kwh",
        "Oficinas": "energia_oficinas_kwh"
    }
    
    datos = []
    for sector_nombre, col in sectores_cols.items():
        consumo_energia = df_energia[col].sum()
        
        # Estimar CO2 y agua proporcional al consumo energético
        proporcion = consumo_energia / df_energia['energia_total_kwh'].sum()
        co2_total_kg = df_energia['co2_kg'].sum() * proporcion
        agua_total_litros = df_energia['agua_litros'].sum() * proporcion
        
        datos.append({
            "sector": sector_nombre,
            "co2_total_toneladas": round(co2_total_kg / 1000, 2),
            "agua_total_m3": round(agua_total_litros / 1000, 2),
            "consumo_energia_kwh": round(consumo_energia, 2),
            "equivalente_arboles": round(co2_total_kg / 21, 0)  # 1 árbol absorbe ~21 kg CO2/año
        })
    
    return {
        "title": "Impacto Ambiental por Sector",
        "data": sorted(datos, key=lambda x: x['co2_total_toneladas'], reverse=True)
    }


def generar_todos_los_graficos():
    """
    Genera todos los gráficos en un solo JSON
    """
    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "graficos": {
            # Gráficos por Sede
            "consumo_total_por_sede": generar_consumo_total_por_sede(),
            "tendencias_consumo_por_sede": generar_tendencias_consumo_por_sede(),
            "eficiencia_por_estudiante": generar_eficiencia_por_estudiante(),
            "emisiones_co2_por_sede": generar_emisiones_co2_por_sede(),
            "consumo_agua_por_sede": generar_consumo_agua_por_sede(),
            
            # Gráficos por Sector
            "consumo_por_sector": generar_consumo_por_sector(),
            "distribucion_consumo_por_sector": generar_distribucion_consumo_por_sector(),
            "tendencias_consumo_por_sector": generar_tendencias_consumo_por_sector(),
            "eficiencia_por_sector_y_sede": generar_eficiencia_por_sector_y_sede(),
            "costos_operacionales_por_sector": generar_costos_operacionales_por_sector(),
            "impacto_ambiental_por_sector": generar_impacto_ambiental_por_sector(),
            
            # Análisis Correlación
            "correlacion_ocupacion_consumo": generar_correlacion_ocupacion_consumo(),
            "temperatura_vs_consumo": generar_temperatura_vs_consumo()
        }
    }


if __name__ == "__main__":
    print("=" * 70)
    print("📊 GENERANDO DATOS PARA GRÁFICOS DE LA API")
    print("=" * 70)
    
    # Generar todos los gráficos
    datos_completos = generar_todos_los_graficos()
    
    # Guardar en archivo JSON
    with open('api_graficos.json', 'w', encoding='utf-8') as f:
        json.dump(datos_completos, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Archivo 'api_graficos.json' generado exitosamente")
    print("\nResumen de gráficos generados:")
    
    for nombre, grafico in datos_completos['graficos'].items():
        print(f"\n📈 {nombre}:")
        print(f"   Título: {grafico.get('title', 'N/A')}")
        if 'data' in grafico:
            if isinstance(grafico['data'], list):
                print(f"   Datos: {len(grafico['data'])} elementos")
            elif isinstance(grafico['data'], dict):
                print(f"   Datos: {len(grafico['data'])} series")
    
    print("\n" + "=" * 70)
    print("✅ GENERACIÓN COMPLETA")
    print("=" * 70)
