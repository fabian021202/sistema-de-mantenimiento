import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

# Configuración de página moderna
st.set_page_config(page_title="Sistema Experto en Mantenimiento Pro", layout="wide")

st.title("🛡️ Sistema Experto de Toma de Decisiones en Mantenimiento v2.0")
st.markdown("---")

# Inicialización de estados de sesión para mantener la consistencia
if 'nombre' not in st.session_state:
    st.session_state.nombre = "Activo Personalizado"
    st.session_state.criticidad = "MEDIA"
    st.session_state.totales = 720.0
    st.session_state.demanda = 600.0
    st.session_state.uptime = 580.0
    st.session_state.fallas = 2
    st.session_state.beta = 1.5
    st.session_state.eta = 1000.0
    st.session_state.adq = 50000.0
    st.session_state.can = 5000.0
    st.session_state.cons = 2500.0
    st.session_state.vida = 5
    st.session_state.fallas_raw = "120, 240, 310, 450, 600"

def cargar_caso(caso_id):
    if caso_id == 1:
        st.session_state.nombre = "Turbocompresor Principal K-101"
        st.session_state.criticidad = "ALTA"
        st.session_state.totales = 720.0
        st.session_state.demanda = 720.0
        st.session_state.uptime = 650.0
        st.session_state.fallas = 2
        st.session_state.beta = 2.8
        st.session_state.eta = 800.0
        st.session_state.adq = 250000.0
        st.session_state.can = 15000.0
        st.session_state.cons = 8000.0
        st.session_state.vida = 6
        st.session_state.fallas_raw = "400, 520, 610, 710, 790"
    elif caso_id == 2:
        st.session_state.nombre = "Bomba de Proceso Standby P-202B"
        st.session_state.criticidad = "MEDIA"
        st.session_state.totales = 720.0
        st.session_state.demanda = 360.0
        st.session_state.uptime = 340.0
        st.session_state.fallas = 1
        st.session_state.beta = 1.0
        st.session_state.eta = 1200.0
        st.session_state.adq = 35000.0
        st.session_state.can = 3000.0
        st.session_state.cons = 1500.0
        st.session_state.vida = 5
        st.session_state.fallas_raw = "1200, 1150, 1250"
    elif caso_id == 3:
        st.session_state.nombre = "Extractor de Aire de Taller E-405"
        st.session_state.criticidad = "BAJA"
        st.session_state.totales = 720.0
        st.session_state.demanda = 100.0
        st.session_state.uptime = 98.0
        st.session_state.fallas = 3
        st.session_state.beta = 0.6
        st.session_state.eta = 400.0
        st.session_state.adq = 5000.0
        st.session_state.can = 400.0
        st.session_state.cons = 50.0
        st.session_state.vida = 4
        st.session_state.fallas_raw = "15, 45, 90, 180"

# Barra Lateral - Carga de Casos
st.sidebar.header("📂 Casos de Estudio")
if st.sidebar.button("Caso 1: Turbocompresor (Alta Criticidad)"):
    cargar_caso(1)
if st.sidebar.button("Caso 2: Bomba Standby (Media Criticidad)"):
    cargar_caso(2)
if st.sidebar.button("Caso 3: Extractor de Aire (Baja Criticidad)"):
    cargar_caso(3)

# Diseño en columnas
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Datos de Operación y Estimación de Weibull")
    nombre = st.text_input("Nombre del Activo", value=st.session_state.nombre)
    criticidad = st.selectbox("Nivel de Criticidad", ["ALTA", "MEDIA", "BAJA"], index=["ALTA", "MEDIA", "BAJA"].index(st.session_state.criticidad))
    
    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        totales = st.number_input("Horas Totales del Periodo", value=st.session_state.totales)
        uptime = st.number_input("Uptime (Horas Operando)", value=st.session_state.uptime)
    with sub_c2:
        demanda = st.number_input("Horas de Demanda Real", value=st.session_state.demanda)
        fallas = st.number_input("Número de Fallas Registradas", value=int(st.session_state.fallas), step=1)
    
    # Nuevo bloque de análisis de datos crudos para Weibull
    st.markdown("**🔧 Estimador de Weibull Integrado**")
    modo_weibull = st.radio("Método para ingresar Beta (β) y Eta (η):", ["Ingreso Manual", "Calcular desde Tiempos de Falla Crudos"])
    
    if modo_weibull == "Ingreso Manual":
        beta = st.number_input("Weibull Beta (β)", value=st.session_state.beta)
        eta = st.number_input("Weibull Eta (η)", value=st.session_state.eta)
    else:
        fallas_raw_input = st.text_area("Tiempos de Falla separados por comas (TBF en horas):", value=st.session_state.fallas_raw)
        try:
            # Algoritmo de Estimación Lineal de Weibull por Rangos Medianos
            datos_tbf = sorted([float(x.strip()) for x in fallas_raw_input.split(",") if x.strip() != ""])
            if len(datos_tbf) >= 2:
                n = len(datos_tbf)
                # Rangos medianos aproximados Benard
                y_median = [(i - 0.3) / (n + 0.4) for i in range(1, n + 1)]
                Y = np.log(np.log(1 / (1 - np.array(y_median))))
                X = np.log(datos_tbf)
                
                # Regresión lineal por mínimos cuadrados
                slope, intercept = np.polyfit(X, Y, 1)
                beta = float(slope)
                eta = float(np.exp(-intercept / slope))
                
                st.success(f"📈 Estimación Exitosa: β = {beta:.2f} | η = {eta:.2f} h")
            else:
                st.warning("Ingrese al menos 2 valores separados por comas para estimar los parámetros.")
                beta, eta = st.session_state.beta, st.session_state.eta
        except Exception as e:
            st.error("Error procesando los datos crudos. Asegúrese de usar números separados por comas.")
            beta, eta = st.session_state.beta, st.session_state.eta

with col2:
    st.subheader("💰 Parámetros Económicos y Ciclo de Vida")
    adq = st.number_input("Costo de Adquisición ($)", value=st.session_state.adq)
    can = st.number_input("Costo Anual de Mantenimiento CAN ($)", value=st.session_state.can)
    cons = st.number_input("Costo de Lucro Cesante ($/Hora de Parada)", value=st.session_state.cons)
    vida = st.number_input("Vida Útil Estimada (Años)", value=int(st.session_state.vida), step=1)

# --- PROCESAMIENTO MATEMÁTICO ---
fallas_calc = max(fallas, 1)
downtime = max(totales - uptime, 0.0)
mtbf = uptime / fallas_calc
mttr = downtime / fallas_calc

disp_inherente = (mtbf / (mtbf + mttr)) * 100 if (mtbf + mttr) > 0 else 0.0
disp_operacional = (uptime / totales) * 100 if totales > 0 else 0.0
factor_utilizacion = uptime / demanda if demanda > 0 else 0.0

costo_riesgo = (1 / mtbf) * cons * mttr * 8760.0
cae_capital = adq / vida if vida > 0 else adq
cae_total = cae_capital + can + costo_riesgo

# Diagnósticos en variables de texto para poder exportarlos después
diag_weibull = ""
diag_disp = ""
diag_cae = ""

if beta < 1:
    diag_weibull = f"DIAGNÓSTICO WEIBULL (β = {beta:.2f}): Mortalidad Infantil. ACCIÓN: Revisar control de calidad en recepción, instalación, alineación geométrico o capacitación del operador. No programar mantenimiento preventivo cíclico aquí."
elif abs(beta - 1.0) <= 0.1:
    diag_weibull = f"DIAGNÓSTICO WEIBULL (β = {beta:.2f}): Fallas Aleatorias. ACCIÓN: Aplicar mantenimiento predictivo avanzado, tareas de monitoreo de condiciones o asegurar stock crítico."
else:
    diag_weibull = f"DIAGNÓSTICO WEIBULL (β = {beta:.2f}): Fase de Desgaste y Envejecimiento. ACCIÓN: Programar Overhaul general inmediato o ejecutar reemplazo absoluto guiado por CAE."

if disp_operacional > 95.0 and factor_utilizacion < 0.40 and criticidad == "BAJA":
    diag_disp = f"ALERTA DE GESTIÓN: Alta Disponibilidad Artificial ({disp_operacional:.1f}%). NOTA EXPERTA: El activo no es robusto, está subutilizado. Evite sobredimensionar planes de mantenimiento."
else:
    diag_disp = "CONGRUENCIA OPERACIONAL: La disponibilidad técnica guarda perfecta relación lógica con la demanda real del proceso."

if criticidad == "ALTA" and costo_riesgo > (cae_capital + can):
    diag_cae = f"ALERTA ECONÓMICA DE RIESGO CRÍTICO: El Costo de Riesgo (${costo_riesgo:,.2f}) desborda los costos operativos básicos. RECOMENDACIÓN FINAL: Bloquear reparaciones paliativas repetitivas. Reemplazar equipo de forma inmediata por obsolescencia."
else:
    diag_cae = "ESTABILIDAD FINANCIERA: El costo de riesgo está controlado dentro de los umbrales de rentabilidad del activo."

# --- RENDERIZADO DE RESULTADOS ---
st.markdown("---")
st.header("📊 Cuadro de Indicadores y Diagnóstico Experto")

m1, m2, m3, m4 = st.columns(4)
m1.metric("MTBF", f"{mtbf:.2f} h")
m2.metric("MTTR", f"{mttr:.2f} h")
m3.metric("Disponibilidad Inherente", f"{disp_inherente:.2f} %")
m4.metric("Disponibilidad Operacional", f"{disp_operacional:.2f} %")

m5, m6 = st.columns(2)
m5.metric("Costo de Riesgo Anualizado", f"${costo_riesgo:,.2f}")
m6.metric("CAE Anualizado Total", f"${cae_total:,.2f}")

st.subheader("🧠 Evaluación Automatizada del Sistema Experto")
if beta < 1: st.error(diag_weibull)
elif abs(beta - 1.0) <= 0.1: st.info(diag_weibull)
else: st.warning(diag_weibull)

if "ALERTA" in diag_disp: st.error(diag_disp)
else: st.success(diag_disp)

if "🚨" in diag_cae or "ALERTA" in diag_cae: st.error(diag_cae)
else: st.success(diag_cae)

# --- BOTÓN DE EXPORTACIÓN EXCEL/CSV ---
st.markdown("### 📥 Reporte de Ingeniería")

datos_reporte = {
    "Concepto / Indicador": [
        "Nombre del Activo", "Criticidad", "MTBF (h)", "MTTR (h)", 
        "Disponibilidad Inherente (%)", "Disponibilidad Operacional (%)", 
        "Costo de Riesgo Anual ($)", "CAE Total ($)", 
        "Diagnóstico Confiabilidad (Weibull)", "Diagnóstico Operacional", "Diagnóstico Económico"
    ],
    "Valor Calculado": [
        nombre, criticidad, round(mtbf, 2), round(mttr, 2), 
        round(disp_inherente, 2), round(disp_operacional, 2), 
        round(costo_riesgo, 2), round(cae_total, 2), 
        diag_weibull, diag_disp, diag_cae
    ]
}

df_reporte = pd.DataFrame(datos_reporte)

# Convertir el DataFrame a un string CSV en memoria
csv_buffer = io.StringIO()
