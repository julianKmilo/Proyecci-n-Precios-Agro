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

# ── CACHÉ DE DATOS ──────────────────────────────────────────────────────────────
@st.cache_data
def load_model_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

dict_modelos = load_model_dict(JSON_PATH)
series_list  = list(dict_modelos.keys())

# ── TÍTULO CENTRADO ─────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align: center;'>📈 Dashboard de Series Agropecuarias</h1>",
    unsafe_allow_html=True
)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header("Controles")

# 1) IRF
irf_files = sorted([f for f in os.listdir(IRF_DIR) if f.lower().endswith((".png", ".jpg"))])
irf_sel   = st.sidebar.selectbox("Función Impulso-Respuesta:", irf_files)

# 2) Selección de una o varias series
selected_series = st.sidebar.multiselect(
    "Series (Forecast + Modelo):",
    series_list,
    default=[series_list[0]]
)

# 3) Rango de fechas (dejar vacío para rango completo)
date_range = st.sidebar.date_input("Rango de fechas:", [])

# ── PÁGINA PRINCIPAL ────────────────────────────────────────────────────────────

# (A) IRF centrada
st.subheader("Función Impulso-Respuesta")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    irf_path = os.path.join(IRF_DIR, irf_sel)
    st.image(irf_path, caption=irf_sel, width=600)

# (B) Estadísticas de bondad de ajuste
st.subheader("🔧 Estadísticas de Bondad de Ajuste")
metrics_list = []
for serie in selected_series:
    info = dict_modelos[serie]
    metrics_list += [
        (f"{serie} RMSE",   float(info.get("rmse", 0))),
        (f"{serie} MAE",    float(info.get("mae", 0))),
        (f"{serie} MAPE",   float(info.get("mape", 0)))
    ]
cols = st.columns(len(metrics_list))
for i, (label, val) in enumerate(metrics_list):
    cols[i].metric(label, f"{val:,.2f}")

# (C) Forecast comparativo
st.subheader("📊 Histórico vs Proyección")
fig = go.Figure()

for serie in selected_series:
    # Cargar cada forecast
    csv_name = serie.replace(" ", "_") + ".csv"
    csv_path = os.path.join(FORECAST_DIR, csv_name)
    if not os.path.exists(csv_path):
        st.warning(f"No se encontró `{csv_name}`")
        continue

    df = pd.read_csv(
        csv_path,
        header=None, skiprows=1,
        usecols=[0,1,2],
        names=["fecha", "historico", "proyeccion"]
    )
    df["fecha"]      = pd.to_datetime(df["fecha"], errors="coerce")
    df["historico"]  = pd.to_numeric(df["historico"], errors="coerce")
    df["proyeccion"] = pd.to_numeric(df["proyeccion"], errors="coerce")
    df = df.dropna(subset=["fecha"]).set_index("fecha").sort_index()

    # Aplicar rango de fechas si hay selección válida
    if len(date_range) == 2:
        start, end = date_range
        df = df.loc[start:end]

    # Añadir trazas al gráfico
    fig.add_trace(go.Scatter(
        x=df.index, y=df["historico"],
        mode="lines", name=f"{serie} (Histórico)"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["proyeccion"],
        mode="lines", name=f"{serie} (Proyección)"
    ))

# Formatear y mostrar
date_label = ""
if len(date_range) == 2:
    start, end = date_range
    date_label = f" ({start:%Y-%m-%d} – {end:%Y-%m-%d})"
fig.update_layout(
    title=f"Comparativo de Series{date_label}",
    xaxis_title="Fecha",
    yaxis_title="Precio",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)
st.plotly_chart(fig, use_container_width=True)

# (D) Detalles del modelo principal
if selected_series:
    primary = selected_series[0]
    with st.expander(f"🛠 Detalles del Modelo para {primary}"):
        st.json(dict_modelos[primary])
