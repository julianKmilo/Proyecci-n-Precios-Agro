import streamlit as st
import pandas as pd
import json, os

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
series_list   = list(dict_modelos.keys())

# ── INTERFAZ ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard Proyecto", layout="wide")
st.title("📈 Dashboard de Series Agropecuarias")

# Sidebar: selección de serie/modelo y de función impulso-respuesta
st.sidebar.header("Controles")

# 1) Serie & Modelo
serie_sel = st.sidebar.selectbox(
    "Selecciona serie (Forecast + Modelo):",
    series_list
)
modelo_info = dict_modelos[serie_sel]

# 2) IRF
irf_files = [f for f in os.listdir(IRF_DIR) if f.lower().endswith((".png",".jpg"))]
irf_sel   = st.sidebar.selectbox("Selecciona IRF:", irf_files)

# ── PÁGINA PRINCIPAL ────────────────────────────────────────────────────────────
# Mostrar IRF
st.subheader("Función Impulso-Respuesta")
irf_path = os.path.join(IRF_DIR, irf_sel)
st.image(irf_path, caption=irf_sel, use_column_width=True)

# Mostrar Forecast
st.subheader(f"Proyección: {serie_sel}")
# Construir nombre de CSV a partir de la clave JSON (e.g. 'aceite de palma' → 'aceite_de_palma.csv')
csv_name = serie_sel.replace(" ", "_") + ".csv"
csv_path = os.path.join(FORECAST_DIR, csv_name)

if os.path.exists(csv_path):
    df_fore = pd.read_csv(csv_path, parse_dates=True)
    # Asume que hay una columna 'fecha' y una de proyección, ajústalas si es necesario
    df_fore = df_fore.sort_values("fecha").set_index("fecha")
    fig = px.line(
        df_fore,
        y=df_fore.columns,
        labels={"value": "Valor", "variable": "Serie"},
        title=f"Forecast de {serie_sel}"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"No se encontró el archivo:\n`{csv_path}`")

# Mostrar detalles del modelo
st.subheader("Parámetros del Modelo Seleccionado")
st.json(modelo_info)
