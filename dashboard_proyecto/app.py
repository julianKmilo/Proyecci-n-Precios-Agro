import streamlit as st
import pandas as pd
import json, os
import plotly.express as px

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
serie_sel   = st.sidebar.selectbox("Selecciona serie (Forecast + Modelo):", series_list)
modelo_info = dict_modelos[serie_sel]
irf_files   = sorted([f for f in os.listdir(IRF_DIR) if f.lower().endswith((".png", ".jpg"))])
irf_sel     = st.sidebar.selectbox("Selecciona IRF:", irf_files)

# ── PÁGINA PRINCIPAL ────────────────────────────────────────────────────────────
# (A) Mostrar IRF
st.subheader("Función Impulso-Respuesta")
irf_path = os.path.join(IRF_DIR, irf_sel)
st.image(irf_path, caption=irf_sel, use_column_width=False, width=600)

# (B) Mostrar Forecast
st.subheader(f"Proyección: {serie_sel}")
csv_name = serie_sel.replace(" ", "_") + ".csv"
csv_path = os.path.join(FORECAST_DIR, csv_name)

if os.path.exists(csv_path):
    # 1) Saltar la fila 0 (serie), 2) leer sólo las dos primeras columnas, 3) nombrarlas
    df_fore = pd.read_csv(
        csv_path,
        header=None,
        skiprows=1,
        usecols=[0, 1],
        names=["fecha", "precio"],
        encoding="utf-8"
    )
    # Parsear y indexar
    df_fore["fecha"] = pd.to_datetime(df_fore["fecha"], errors="coerce")
    df_fore["precio"] = pd.to_numeric(df_fore["precio"], errors="coerce")
    df_fore = df_fore.dropna(subset=["fecha"]) \
                     .set_index("fecha") \
                     .sort_index()

    # Graficar
    fig = px.line(
        df_fore,
        y="precio",
        labels={"fecha": "Fecha", "precio": "Valor"},
        title=f"Forecast de {serie_sel}"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning(f"No se encontró el archivo de forecast:\n`{csv_path}`")

# (C) Parámetros del Modelo
st.subheader("Parámetros del Modelo Seleccionado")
st.json(modelo_info)
