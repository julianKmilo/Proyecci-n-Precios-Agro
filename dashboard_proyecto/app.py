import streamlit as st
import pandas as pd
import json, os
import plotly.graph_objects as go

# ── RUTAS ───────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
DATA_DIR      = os.path.join(BASE_DIR, "data")
IRF_DIR       = os.path.join(DATA_DIR, "IRF")
FORECAST_DIR  = os.path.join(DATA_DIR, "Forecast")
JSON_PATH     = os.path.join(DATA_DIR, "dict_modelos.json")

# ── CARGAR DICT DE MODELOS ─────────────────────────────────────────────────────
@st.cache(allow_output_mutation=True)
def load_model_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

dict_modelos = load_model_dict(JSON_PATH)
series_list  = list(dict_modelos.keys())

# ── CONFIGURACIÓN PÁGINA ───────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard Proyecto", layout="wide")
st.title("📈 Dashboard de Series Agropecuarias")

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header("Controles")

# 1) Serie & Modelo
serie_sel   = st.sidebar.selectbox("Serie (Forecast + Modelo):", series_list)
modelo_info = dict_modelos[serie_sel]

# 2) IRF (podrías filtrar por variables de modelo si dict_modelos lo permite)
irf_files   = sorted([f for f in os.listdir(IRF_DIR) if f.lower().endswith((".png", ".jpg"))])
irf_sel     = st.sidebar.selectbox("Función Impulso-Respuesta:", irf_files)

# 3) Rango de fechas para forecast
# (más tarde inicializaremos con los límites reales del DF)
date_range = st.sidebar.date_input("Rango de fechas:", [])

# ── PÁGINA PRINCIPAL ────────────────────────────────────────────────────────────

# (A) IRF
st.subheader("📷 Función Impulso-Respuesta")
irf_path = os.path.join(IRF_DIR, irf_sel)
st.image(irf_path, caption=irf_sel, use_column_width=False, width=600)

# (B) Forecast
st.subheader(f"📊 Proyección de {serie_sel}")
csv_name = serie_sel.replace(" ", "_") + ".csv"
csv_path = os.path.join(FORECAST_DIR, csv_name)

if os.path.exists(csv_path):
    # Leer 3 columnas: fecha / histórico / proyección
    df = pd.read_csv(
        csv_path,
        header=None,
        skiprows=1,
        usecols=[0, 1, 2],
        names=["fecha", "historico", "proyeccion"]
    )
    df["fecha"]      = pd.to_datetime(df["fecha"], errors="coerce")
    df["historico"]  = pd.to_numeric(df["historico"], errors="coerce")
    df["proyeccion"] = pd.to_numeric(df["proyeccion"], errors="coerce")
    df = df.dropna(subset=["fecha"]).set_index("fecha").sort_index()
    
    # Si el usuario no seleccionó rango, lo definimos completo
    if not date_range or len(date_range) != 2:
        start, end = df.index.min(), df.index.max()
    else:
        start, end = date_range
    df = df.loc[start:end]
    
    # Métricas clave
    col1, col2, col3 = st.columns(3)
    col1.metric("Último Histórico", f"{df['historico'].iloc[-1]:,.2f}")
    col2.metric("Primer Proyección", f"{df['proyeccion'].iloc[0]:,.2f}")
    cambio = df['proyeccion'].iloc[-1] - df['historico'].iloc[-1]
    col3.metric("Cambio Proyectado", f"{cambio:,.2f}")

    # Chart comparativo
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["historico"],
        mode="lines", name="Histórico"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["proyeccion"],
        mode="lines", name="Proyección"
    ))
    fig.update_layout(
        title=f"Histórico vs Proyección ({start.date()} – {end.date()})",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Botón para descargar datos filtrados
    csv_download = df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar CSV",
        data=csv_download,
        file_name=f"{serie_sel}_forecast_{start.date()}_{end.date()}.csv",
        mime="text/csv"
    )
else:
    st.warning(f"No se encontró el archivo:\n`{csv_path}`")

# (C) Detalles del modelo
with st.expander("🛠 Detalles del Modelo Seleccionado"):
    st.json(modelo_info)

# (D) Sugerencia de mejoras adicionales
st.markdown("""
**Ideas para enriquecer aún más el dashboard**  
- Filtrar IRF por variables macroeconómicas asociadas al modelo (si `modelo_info` incluye dicha lista).  
- Mostrar estadísticas de bondad de ajuste (RMSE, MAE, MAPE) en un panel de KPIs.  
- Agregar un selector para superponer varias series de forecast y compararlas.  
- Incluir un mapa o gráfico de barras si tus datos cubren varias regiones o productos.  
- Añadir alertas automáticas si la proyección supera ciertos umbrales (por ejemplo, `st.toast` o un badge).  
""")
