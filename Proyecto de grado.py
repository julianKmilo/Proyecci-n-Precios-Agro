# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.stats import kruskal
from scipy.stats import norm
from numpy.linalg import LinAlgError, pinv
from itertools import chain
import dateparser
import textwrap

import statsmodels.api as sm
from statsmodels.tsa.stattools      import adfuller, kpss
from statsmodels.stats.diagnostic   import acorr_ljungbox, het_arch
from statsmodels.stats.stattools    import jarque_bera
from statsmodels.tsa.vector_ar.var_model  import VAR
from statsmodels.tsa.vector_ar.vecm       import coint_johansen, VECM, select_coint_rank
from statsmodels.formula.api        import ols
from statsmodels.tsa.seasonal       import seasonal_decompose
from statsmodels.tsa.seasonal import STL


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics        import mean_squared_error, mean_absolute_error

import pygad  # pip install pygad
from tensorflow.keras          import Input, Model
from tensorflow.keras.layers   import SimpleRNN, Conv1D, GRU, Dense
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks  import EarlyStopping

from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.graph_objects as go

from PIL import Image, ImageDraw, ImageFont

# %%
def df_to_image_optimized(df, filename='dataframe_optimized.png', 
                         font_size=12, max_col_width=300, min_col_width=50):
    """
    Convierte un DataFrame a imagen con formato profesional
    
    Args:
        df: DataFrame de pandas (9x4 en tu caso)
        filename: Nombre del archivo de salida
        font_size: Tamaño base de fuente (se ajusta automáticamente)
        max_col_width: Ancho máximo por columna en píxeles
        min_col_width: Ancho mínimo por columna en píxeles
    """
    # Configuración inicial
    padding_x = 15  # Espaciado horizontal interno
    padding_y = 10  # Espaciado vertical interno
    line_thickness = 1  # Grosor de líneas de separación
    
    # Convertir datos a strings
    df_display = df.copy().astype(str)
    cols = df_display.columns.tolist()
    arr = df_display.values
    
    # Calcular anchos de columna basados en contenido
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Función para calcular ancho de texto
    def get_text_width(text, font):
        return font.getlength(str(text))
    
    col_widths = []
    for j in range(arr.shape[1]):
        # Ancho máximo del contenido
        content_width = max([get_text_width(arr[i,j], font) for i in range(arr.shape[0])] + [get_text_width(cols[j], font)])
        # Ajustar entre mínimo y máximo
        col_width = max(min_col_width, min(max_col_width, content_width + 2*padding_x))
        col_widths.append(int(col_width))
    
    # Calcular altos de fila
    row_height = font_size + 2*padding_y
    
    # Crear imagen
    img_width = sum(col_widths) + (len(col_widths)-1)*line_thickness
    img_height = row_height*(arr.shape[0]+1) + arr.shape[0]*line_thickness
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Dibujar cabeceras
    x_pos = 0
    for j, col in enumerate(cols):
        draw.rectangle([x_pos, 0, x_pos+col_widths[j], row_height], fill='#2c7be5')
        draw.text((x_pos+padding_x, padding_y), col, fill='white', font=font)
        x_pos += col_widths[j] + line_thickness
    
    # Dibujar datos
    y_pos = row_height + line_thickness
    for i in range(arr.shape[0]):
        x_pos = 0
        for j in range(arr.shape[1]):
            # Color de fondo alternado
            fill_color = '#ffffff' if i % 2 == 0 else '#f9fafc'
            draw.rectangle([x_pos, y_pos, x_pos+col_widths[j], y_pos+row_height], fill=fill_color)
            
            # Alineación derecha para números, izquierda para texto
            text = arr[i,j]
            text_width = get_text_width(text, font)
            
            if df.iloc[:,j].dtype.kind in 'iuf':  # Números
                text_x = x_pos + col_widths[j] - padding_x - text_width
            else:  # Texto
                text_x = x_pos + padding_x
            
            draw.text((text_x, y_pos+padding_y), text, fill='#2d3748', font=font)
            
            # Bordes verticales
            if j < arr.shape[1]-1:
                draw.line([x_pos+col_widths[j], y_pos, x_pos+col_widths[j], y_pos+row_height], 
                         fill='#e5e7eb', width=line_thickness)
            
            x_pos += col_widths[j] + line_thickness
        
        # Borde horizontal inferior
        draw.line([0, y_pos+row_height, img_width, y_pos+row_height], 
                 fill='#e5e7eb', width=line_thickness)
        y_pos += row_height + line_thickness
    
    # Bordes exteriores
    draw.rectangle([0, 0, img_width-1, img_height-1], outline='#d1d9e6', width=2)
    
    img.save(filename, dpi=(300, 300), quality=95)
    print(f"✅ Imagen guardada como {filename}")


# %% [markdown]
# # 1 Analisis exploratorio

# %%
# 1. Carga de variables macro

df_m = pd.read_excel('C:/Users/USER/Documents/Proyecto de Grado/Datos AGRO.xlsx', sheet_name='MACRO', parse_dates=['Fecha'])
df_q = pd.read_excel('C:/Users/USER/Documents/Proyecto de Grado/Datos AGRO.xlsx', sheet_name='MACRO2', parse_dates=['Fecha'])

# 1. Convertir 'Fecha' a periodos mensuales y usarlo como índice
df_m['Mes'] = df_m['Fecha'].dt.to_period('M').dt.to_timestamp()
df_wide = df_m.set_index('Mes', drop=True)

df_q['Trimestre'] = df_q['Fecha'].dt.to_period('Q').dt.to_timestamp()
df_q = df_q.set_index('Trimestre', drop=True)


# 2. Eliminar la columna original si ya no es necesaria
df_wide = df_wide.drop(columns=['Fecha'], errors='ignore')
df_q = df_q.drop(columns=['Fecha'], errors='ignore')


# 3. Reindexar para asegurar continuidad mensual
idx = pd.date_range(df_wide.index.min(), df_wide.index.max(), freq='MS')
df_wide = df_wide.reindex(idx)
df_wide.index.name = 'Mes'

idx_qm = pd.date_range(df_q.index.min(), df_q.index.max(), freq='MS')
monthly_from_q = df_q.reindex(idx_qm, method='ffill')
monthly_from_q.index.name = 'Mes'


# 4. Rellenar únicamente huecos internos por interpolación lineal
df_wide = df_wide.interpolate(method='time', limit_area='inside')

# 5. Construir un diccionario de DataFrames por variable
dict_macro_m = {
    var: df_wide[[var]].dropna()
    for var in df_wide.columns
}

dict_macro_t = {
    var: monthly_from_q[[var]].dropna()
    for var in monthly_from_q.columns
}

dict_macro = dict_macro_m | dict_macro_t

# %%
# 1. Cargar los dos sheets
df1 = pd.read_excel(
    'C:/Users/USER/Documents/Proyecto de Grado/Datos AGRO.xlsx',
    sheet_name='Precios AGRO',
    parse_dates=['Fecha']
)
df2 = pd.read_excel(
    'C:/Users/USER/Documents/Proyecto de Grado/Datos AGRO.xlsx',
    sheet_name='Precios AGRO2',
    parse_dates=['Fecha']
)

df1['Producto'] = (df1['Producto'].astype(str).str.lower())

# --- PROCESAMIENTO df1 (Precios AGRO) ---
# 2. Convertir a periodo mensual y timestamp
df1['Mes'] = df1['Fecha'].dt.to_period('M').dt.to_timestamp()
# 3. Precio promedio mensual
monthly_avg1 = (
    df1
    .groupby(['Mes', 'Producto'])['Precio x kilo']
    .mean()
    .reset_index()
)
# 4. Pivotar
price_ts1 = monthly_avg1.pivot(index='Mes', columns='Producto', values='Precio x kilo')
# 5. Reindexar para no perder meses
idx1 = pd.date_range(price_ts1.index.min(), price_ts1.index.max(), freq='MS')
price_ts1 = price_ts1.reindex(idx1).rename_axis('Mes')
# 6. Rellenar huecos internos
price_ts1 = price_ts1.interpolate(method='time', limit_area='inside')
# 7. Diccionario parcial
dict_productos1 = {
    prod: price_ts1[[prod]].dropna()
    for prod in price_ts1.columns
}

# --- PROCESAMIENTO df2 (Precios AGRO2) ---
# 2. Parsear fecha y convertir a periodo mensual
df2['Mes'] = (
    df2['Fecha']
    .astype(str)
    .apply(lambda x: dateparser.parse(x, languages=['es']))
    .dt.to_period('M')
    .dt.to_timestamp()
)
# 3. Agrupar y pivotar
resumen2 = (
    df2
    .groupby(['Mes', 'Producto'], as_index=False)['Precio']
    .mean()
)
price_ts2 = resumen2.pivot(index='Mes', columns='Producto', values='Precio')

# 4. Cargar TRM mensual (debe existir dict_macro con la serie)
df_trm = dict_macro['TRM'].reset_index()
df_trm['Mes'] = df_trm['Mes'].dt.to_period('M').dt.to_timestamp()
trm_ts = df_trm.set_index('Mes')

# 5. Definir índice común y reindexar ambas
start = min(price_ts2.index.min(), trm_ts.index.min())
end   = max(price_ts2.index.max(), trm_ts.index.max())
idx2 = pd.date_range(start, end, freq='MS')

price_ts2 = price_ts2.reindex(idx2).rename_axis('Mes')
trm_ts   = trm_ts  .reindex(idx2).rename_axis('Mes')

# 6. Rellenar huecos internos
price_ts2 = price_ts2.interpolate(method='time', limit_area='inside')
trm_ts    = trm_ts   .interpolate(method='time', limit_area='inside')

# 7. Convertir precios a pesos
price_ts2_cop = price_ts2.multiply(
    trm_ts['TRM'],
    axis=0
) / 100

# 8. Diccionario parcial 2
dict_productos2 = {
    prod: price_ts2_cop[[prod]].dropna()
    for prod in price_ts2_cop.columns
}

# --- UNIFICAR AMBOS DICCIONARIOS ---
dict_productos = { **dict_productos1, **dict_productos2 }



# %%
len(dict_productos)

# %%
# 1. Filtrar dict_productos
keys_to_keep = [
    "pollo entero fresco sin vísceras",
    "carne de cerdo, lomo sin hueso",
    "carne de res, lomo de brazo",
    "arroz de segunda",
    "Azúcar 11 (Crudo)",
    "Café",
    "Algodón",
    "Soya",
    "Cacao Nacional"
]
dict_productos_filt = {k: dict_productos[k] for k in keys_to_keep if k in dict_productos}

# Asegurarnos de que el índice de cada DataFrame/Series sea datetime
for d in dict_productos_filt.values():
    d.index = pd.to_datetime(d.index)
for d in dict_macro.values():
    d.index = pd.to_datetime(d.index)

# 2. Scatter plots: cada serie agro vs cada serie macro
for agro_name, agro_df in dict_productos_filt.items():
    # Extraer la serie de precios (primera columna si es DataFrame)
    serie_agro = agro_df.iloc[:, 0] if isinstance(agro_df, pd.DataFrame) else agro_df
    serie_agro = pd.to_numeric(serie_agro, errors='coerce').dropna().rename(agro_name)
    
    # Preparamos las series macro, asegurándonos de que sean numéricas
    macro_series = {
        m_name: pd.to_numeric(
            (m_df.iloc[:,0] if isinstance(m_df, pd.DataFrame) else m_df),
            errors='coerce'
        ).dropna().rename(m_name)
        for m_name, m_df in dict_macro.items()
    }
    
    # Número de variables macro y diseño de subplots
    num_macros = len(macro_series)
    cols = 3
    rows = math.ceil(num_macros / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=False, sharey=False)
    axes = axes.flatten()
    
    # Título general
    fig.suptitle(f"{agro_name} VS", fontsize=16, y=1.02)
    
    # Crear cada scatter en su propio subplot
    for ax, (m_name, m_ser) in zip(axes, macro_series.items()):
        # Unir las dos series en base a fechas comunes
        df_pair = pd.concat([serie_agro, m_ser], axis=1, join='inner').dropna()
        
        ax.scatter(df_pair[m_name].values, df_pair[agro_name].values)
        ax.set_title(m_name, fontsize=10)
        ax.set_xlabel(m_name, fontsize=8)
        ax.set_ylabel(agro_name, fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
    
    # Desactivar ejes sobrantes si macro_series < filas*columnas
    for ax in axes[num_macros:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# %%
# 3. Gráfico de líneas de los precios agro vs tiempo

# Convertimos dict a lista de tuplas para iterar
agro_items = list(dict_productos_filt.items())

# Creamos una figura con 3 filas y 3 columnas
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), sharex=True)

# Aplanamos la matriz de ejes para recorrerlos fácilmente
axes_flat = axes.flatten()

for ax, (agro_name, agro_df) in zip(axes_flat, agro_items):
    # Extraemos la serie (primera columna) si es DataFrame
    serie = agro_df.iloc[:, 0] if isinstance(agro_df, pd.DataFrame) else agro_df
    
    ax.plot(serie.index, serie.values)
    ax.set_title(agro_name, fontsize=10)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# Si hubiera más ejes que series (no en este caso), los desactivamos
for ax in axes_flat[len(agro_items):]:
    ax.axis('off')

plt.tight_layout()
plt.show()


# %%
# 4. Boxplot de precios agro en una cuadrícula de 3 columnas, cada uno con escala independiente
items = list(dict_productos_filt.items())
n = len(items)
cols = 3
rows = math.ceil(n / cols)

# Quitamos el compartir ejes para que cada subplot ajuste su propia escala
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 4), sharey=False)
axes_flat = axes.flatten()

for ax, (agro_name, agro_df) in zip(axes_flat, items):
    # Extraer la serie (primera columna) si es DataFrame
    serie = agro_df.iloc[:, 0] if isinstance(agro_df, pd.DataFrame) else agro_df
    ax.boxplot(serie.values, whis=1.5)
    ax.set_title(agro_name, fontsize=10)
    ax.set_ylabel("Precio")
    ax.tick_params(axis='x', rotation=45, labelsize=8)

# Desactivar ejes sobrantes si los hay
for ax in axes_flat[n:]:
    ax.axis('off')

plt.tight_layout()
plt.show()


# %%
#5. Matriz de correlaciones
# 1) Convertir dict_productos_filt en un DataFrame de precios agro
df_agro_all = pd.DataFrame({
    k: (v.iloc[:, 0] if isinstance(v, pd.DataFrame) else v)
    for k, v in dict_productos_filt.items()
})

# 2) Convertir dict_macro en un DataFrame de variables macro
df_macro_all = pd.DataFrame({
    k: (v.iloc[:, 0] if isinstance(v, pd.DataFrame) else v)
    for k, v in dict_macro.items()
})

# 3) Concatenar ambos DataFrames usando sólo fechas comunes
df_comb = pd.concat([df_agro_all, df_macro_all], axis=1, join='inner')

# 4) Asegurarse de que todo sea numérico (convierte '-' u otros textos a NaN)
df_comb = df_comb.apply(pd.to_numeric, errors='coerce')

# 5) Eliminar columnas completamente vacías y filas con cualquier NaN
df_comb = df_comb.dropna(axis=1, how='all').dropna()

# 6) Calcular la matriz de correlaciones
corr_matrix = df_comb.corr()


labels = corr_matrix.columns.tolist()

# 1) Envolver el texto en cada etiqueta (máx. 15 caracteres por línea)
wrapped = [
    "\n".join(textwrap.wrap(label, width=15))
    for label in labels
]

# 2) Crear figura mayor y ajustar márgenes
fig, ax = plt.subplots(figsize=(12, 10))

# Opcional: enmascarar triángulo superior para claridad
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 3) Mostrar la matriz con imshow
cax = ax.imshow(
    np.ma.array(corr_matrix.values, mask=mask),
    cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest'
)
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

# 4) Ajustar ticks con las etiquetas envueltas
ax.set_xticks(np.arange(len(wrapped)))
ax.set_yticks(np.arange(len(wrapped)))
ax.set_xticklabels(wrapped, rotation=45, ha='right', fontsize=5)
ax.set_yticklabels(wrapped, fontsize=5)

# 5) Anotar cada casilla (solo el triángulo inferior)
for i in range(len(labels)):
    for j in range(len(labels)):
        if not mask[i, j]:
            val = corr_matrix.iat[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(
                j, i, f"{val:.2f}",
                ha='center', va='center',
                color=color, fontsize=6
            )

# 6) Ajustar el layout para que no se corten las etiquetas
plt.subplots_adjust(bottom=0.3, left=0.3)
#ax.set_title("Tabla de Correlaciones", pad=20, fontsize=14)
plt.show()


# %%
# 6. Histogramas
# Lista de todas las variables en df_comb
vars_all = df_macro_all.columns.tolist()
n = len(vars_all)
cols = 3
rows = math.ceil(n / cols)

# Crear la figura y grid de subplots
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 4), sharex=False, sharey=False)
axes_flat = axes.flatten()

# Título general
#fig.suptitle("Histogramas de todas las variables", fontsize=16, y=1.02)

# Graficar cada histograma en su subplot
for ax, var in zip(axes_flat, vars_all):
    data = df_comb[var].dropna()
    ax.hist(data, bins=20, edgecolor='black')
    ax.set_title(var, fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=8)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

# Desactivar ejes sobrantes si hay más celdas que variables
for ax in axes_flat[n:]:
    ax.axis('off')

plt.tight_layout()
plt.show()


# %%
# 7. Descomposición estacional de los precios agro (series mensuales)

# Lista de productos (hasta 9)
productos = list(dict_productos_filt.keys())[:9]
n = len(productos)

# Parámetros de la cuadrícula
n_cols = 3
n_sets = math.ceil(n / n_cols)      # bloques de 3 productos
total_rows = n_sets * 4             # 4 filas (componentes) por bloque

# Crear figura y ejes
fig, axes = plt.subplots(total_rows, n_cols, 
                         figsize=(n_cols*5, total_rows*2),
                         sharex='col')
fig.subplots_adjust(top=0.95, hspace=0.4)
fig.suptitle('Descomposición Estacional de Precios Agro (9 productos)', 
             y=0.99, fontsize=16)

# Iterar y rellenar cada subplot
for idx, producto in enumerate(productos):
    bloque = idx // n_cols
    col    = idx % n_cols
    df     = dict_productos_filt[producto]
    
    # Índice datetime
    df.index = pd.to_datetime(df.index)
    serie    = df[producto].fillna(method='ffill').fillna(method='bfill')
    descomp  = seasonal_decompose(
        serie, model='additive', period=12, extrapolate_trend='freq'
    )
    
    # Fila base para este bloque
    base_row = bloque * 4
    
    # 1) Original
    ax0 = axes[base_row + 0, col]
    ax0.plot(serie)
    ax0.set_ylabel('Original')
    ax0.set_title(producto, pad=8)
    
    # 2) Tendencia
    ax1 = axes[base_row + 1, col]
    ax1.plot(descomp.trend)
    ax1.set_ylabel('Tendencia')
    
    # 3) Componente estacional
    ax2 = axes[base_row + 2, col]
    ax2.plot(descomp.seasonal)
    ax2.set_ylabel('Estacional')
    
    # 4) Residuo (puntos)
    ax3 = axes[base_row + 3, col]
    ax3.scatter(descomp.resid.index, descomp.resid, s=10, alpha=0.7)
    ax3.set_ylabel('Residuo')
    ax3.set_xlabel('Fecha')

# Eliminar ejes sobrantes si hay menos de 9 productos
for extra in range(n, n_sets * n_cols):
    bloque = extra // n_cols
    col    = extra % n_cols
    for r in range(4):
        fig.delaxes(axes[bloque*4 + r, col])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %% [markdown]
# # 2. Modelo VAR

# %%
def make_stationary(series, significance=0.05, max_diffs=2): # Se usa dentro de ajuste_estacionareidad
    """
    Diferencia la serie hasta lograr estacionariedad según ADF.
    Devuelve la serie diferenciada y el número de diferencias aplicadas.
    """
    d = 0
    s = series.copy()
    try:
        p = adfuller(s.dropna())[1]
    except ValueError:
        return s, 0  # no difiere si no se puede testear
    while p > significance and d < max_diffs:
        s = s.diff().dropna()
        d += 1
        try:
            p = adfuller(s)[1]
        except ValueError:
            break
    return s, d

def ajuste_estacionareidad(dict_productos):
    dict_estacionarias = {}
    lista_rezagos = []
    
    for producto, df_prod in dict_productos.items():
        raw = df_prod[producto].dropna()
        # Omitir constantes y series muy cortas
        if raw.nunique() <= 1:
            print(f"{producto}: Serie constante. Se omite.")
            lista_rezagos.append({'Producto': producto, 'Rezagos': None, 'Estacionaria': False, 'Razon': 'Serie constante'})
            continue
        try:
            p0 = adfuller(raw)[1]
        except ValueError:
            print(f"{producto}: Serie muy corta para ADF inicial. Se omite.")
            lista_rezagos.append({'Producto': producto, 'Rezagos': None, 'Estacionaria': False, 'Razon': 'Serie muy corta para ADF inicial'})
            continue

        # Si ya es estacionaria, la guardamos tal cual
        if p0 < 0.05:
            print(f"{producto}: ya estacionaria (p={p0:.3f}).")
            dict_estacionarias[producto] = raw.to_frame(name=producto)
            lista_rezagos.append({'Producto': producto, 'Rezagos': 0, 'Estacionaria': True, 'Razon': f'Ya estacionaria (p={p0:.3f})'})
            continue

        # Si no, intentamos diferenciar
        stat, d = make_stationary(raw)
        if d == 0:
            print(f"{producto}: no logró estacionar (p inicial={p0:.3f}). Se omite.")
            lista_rezagos.append({'Producto': producto, 'Rezagos': 0, 'Estacionaria': False, 'Razon': f'No se logró estacionar (p inicial={p0:.3f})'})
            continue

        # Verificamos ADF tras diferenciar
        try:
            p1 = adfuller(stat)[1]
        except ValueError:
            print(f"{producto}: diferenciada {d} veces, pero serie muy corta para ADF final. Se omite.")
            lista_rezagos.append({'Producto': producto, 'Rezagos': d, 'Estacionaria': False, 'Razon': 'Serie muy corta para ADF tras diferenciación'})
            continue

        if p1 < 0.05:
            print(f"{producto}: diferenciada {d} veces → estacionaria (p={p1:.3f}).")
            dict_estacionarias[producto] = stat.to_frame(name=producto)
            lista_rezagos.append({'Producto': producto, 'Rezagos': d, 'Estacionaria': True, 'Razon': f'Estacionaria con {d} rezagos (p={p1:.3f})'})
        else:
            print(f"{producto}: diferenciada {d} veces, pero sigue no estacionaria (p={p1:.3f}). Se omite.")
            lista_rezagos.append({'Producto': producto, 'Rezagos': d, 'Estacionaria': False, 'Razon': f'Sigue no estacionaria con {d} rezagos (p={p1:.3f})'})

    # Crear DataFrame con la información de rezagos
    df_rezagos = pd.DataFrame(lista_rezagos)
    
    return dict_estacionarias, df_rezagos

def adjust_seasonality(dict_series, period=12, alpha=0.05, model_type='auto'):
    """
    Ajusta estacionalidad y retorna todas las series (ajustadas y originales) + reporte
    
    Parámetros:
    -----------
    dict_series : dict
        Diccionario {nombre: pd.DataFrame} con series en columnas
    period : int
        Periodicidad de los datos (12 para mensual, 4 para trimestral)
    alpha : float
        Nivel de significancia para pruebas estadísticas
    model_type : str
        'additive', 'multiplicative' o 'auto' para selección automática
    
    Retorna:
    --------
    tuple: (dict_all_series, df_report)
        - dict_all_series: Diccionario con todas las series (ajustadas y originales)
        - df_report: DataFrame con diagnóstico detallado
    """
    dict_all_series = {}
    report_data = []
    
    for name, df in dict_series.items():
        entry = {
            'series': name,
            'status': 'original',
            'model': None,
            'p_value': None,
            'seasonal_strength': None,
            'obs': len(df[name].dropna()),
            'error': None
        }
        
        try:
            # Preprocesamiento común a todas las series
            series = df[name].ffill().dropna()
            dict_all_series[name] = series.to_frame(name=name)  # Guardar original preprocesado
            
            if len(series) < 3 * period:
                entry.update({
                    'status': 'insufficient_data',
                    'error': f"Requires {3*period} observations"
                })
                report_data.append(entry)
                continue
                
            # Selección de modelo
            if model_type == 'auto':
                model = 'multiplicative' if (series > 0).all() and (series.diff().dropna()/series.shift(1).dropna()).std() > 0.2 else 'additive'
            else:
                model = model_type
            entry['model'] = model
            
            # Test de estacionalidad
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                groups = [series[series.index.month == m+1] for m in range(period)]
                _, pval = kruskal(*[g for g in groups if len(g) > 0])
            entry['p_value'] = pval
            
            if pval >= alpha:
                report_data.append(entry)
                continue  # Mantener serie original
                
            # Descomposición y ajuste
            try:
                if len(series) > 100:
                    decomp = STL(series, period=period, robust=True).fit()
                    seasonal = decomp.seasonal
                else:
                    decomp = seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')
                    seasonal = decomp.seasonal
                
                # Aplicar ajuste y actualizar diccionario
                adj_series = series / (seasonal + 1e-10) if model == 'multiplicative' else series - seasonal
                dict_all_series[name] = adj_series.to_frame(name=name)
                
                # Actualizar reporte
                strength = seasonal.std()/series.std()
                entry.update({
                    'status': 'adjusted',
                    'seasonal_strength': strength
                })
                
            except Exception as e:
                entry.update({
                    'status': 'adjustment_failed',
                    'error': str(e)
                })
                
        except Exception as e:
            entry.update({
                'status': 'processing_error',
                'error': str(e)
            })
        
        report_data.append(entry)
    
    # Crear DataFrame de reporte
    df_report = pd.DataFrame(report_data)
    status_order = [
        'adjusted', 'original', 'insufficient_data', 
        'adjustment_failed', 'processing_error'
    ]
    df_report['status'] = pd.Categorical(df_report['status'], categories=status_order, ordered=True)
    
    # Reporte consolidado
    print("=== Seasonality Adjustment Report ===")
    print(f"Total series: {len(dict_series)}")
    print(f"Adjusted: {len(df_report[df_report['status'] == 'adjusted'])}")
    print(f"Non-seasonal: {len(df_report[df_report['status'] == 'original'])}")
    print(f"Failures: {len(df_report[df_report['status'].isin(['adjustment_failed','processing_error'])])}")
    
    return dict_all_series, df_report.sort_values('status')

def select_var_lags(dict_agro, dict_macro, maxlags=12):
    """
    Para cada pareja (producto_agro, variable_macro):
      - alinea sus fechas,
      - intenta select_order con maxlags decrecientes hasta que funcione,
      - guarda el lag elegido por AIC y BIC.
    
    Devuelve:
      - dict { (agro, macro): {'aic': , 'bic': , 'used_lag': } }
      - DataFrame con todos los resultados y metadatos
    """
    lag_selection = {}
    results = []

    for agro, df_agro in dict_agro.items():
        s_agro = df_agro[agro].dropna()
        for macro, df_mac in dict_macro.items():
            s_mac = df_mac[macro].dropna()
            
            # Registro base para el DataFrame
            entry = {
                'agro': agro,
                'macro': macro,
                'n_obs': 0,
                'maxlags_tested': 0,
                'aic': None,
                'bic': None,
                'used_lag': None,
                'status': 'no_data'
            }

            try:
                # Alinear series temporales
                df_pair = pd.concat([s_agro, s_mac], axis=1, join='inner').dropna()
                df_pair.columns = ['y1', 'y2']
                n_obs = len(df_pair)
                entry['n_obs'] = n_obs

                if n_obs < 3:
                    entry['status'] = 'insufficient_data'
                    results.append(entry)
                    continue

                # Rango de lags a probar
                max_to_test = min(maxlags, n_obs - 1)
                entry['maxlags_tested'] = max_to_test
                chosen = None

                for p in range(max_to_test, 0, -1):
                    try:
                        var_model = VAR(df_pair)
                        sel = var_model.select_order(p)
                        
                        # Guardar en diccionario
                        lag_selection[(agro, macro)] = {
                            'aic': sel.aic,
                            'bic': sel.bic,
                            'used_lag': p
                        }
                        
                        # Actualizar entrada del DataFrame
                        entry.update({
                            'aic': sel.aic,
                            'bic': sel.bic,
                            'used_lag': p,
                            'status': 'success'
                        })
                        
                        chosen = p
                        break
                    
                    except (ValueError, np.linalg.LinAlgError) as e:
                        if p == 1:  # Último intento fallido
                            entry['status'] = 'no_valid_lags'
                        continue

                if not chosen:
                    entry['status'] = 'all_lags_failed'

            except Exception as e:
                entry['status'] = f'error: {str(e)[:30]}'

            results.append(entry)

    # Crear DataFrame con tipos adecuados
    df_results = pd.DataFrame(results).astype({
        'n_obs': 'Int64',
        'maxlags_tested': 'Int64',
        'used_lag': 'Int64'
    })

    return lag_selection, df_results



def plot_macro_to_agro_irf_grid(
    dict_agro, dict_macro, lag_selection, steps=12, orth=True, alpha=0.05
):
    """
    Para cada par (agro, macro):
      1) Ajusta VAR(p) con p tomado de lag_selection[(agro,macro)]['aic'] o ['bic']
      2) Extrae la IRF de macro→agro, calcula banda de confianza al 1−alpha
      3) Si en h=1 el intervalo no cruza cero, 
         • dibuja la IRF,
         • almacena en el DataFrame: agro, macro, señal del choque en h=1.
    Devuelve ese DataFrame de IRFs significativas.
    """
    z = norm.ppf(1 - alpha/2)
    records = []

    for (agro, macro), params in lag_selection.items():
        p = params.get('aic', params.get('bic'))
        # 1) Construir y limpiar las series
        s_ag = dict_agro[agro][agro].dropna()
        s_ma = dict_macro[macro][macro].dropna()
        df = pd.concat([s_ag, s_ma], axis=1, join='inner').dropna()
        df.columns = ['y_ag', 'y_ma']
        if len(df) < p + 2:
            continue

        # 2) Ajustar el VAR y extraer IRF y covarianzas
        try:
            model = VAR(df).fit(p)
            irf_res = model.irf(steps)
        except Exception:
            continue

        irfs = irf_res.irfs             # (steps+1, 2, 2)
        covs = irf_res.cov(orth)        # (steps+1, 4, 4)
        resp = irfs[:, 0, 1]
        errs = np.array([np.sqrt(covs[h, 1, 1]) for h in range(len(resp))])

        lower = resp - z * errs
        upper = resp + z * errs

        # 3) Chequear significancia en h=1
        if np.isnan(lower[1]) or not ((lower[1] > 0) or (upper[1] < 0)):
            continue

        # 4) Guardar en el registro la señal del choque en h=1
        sign = 'positive' if resp[1] > 0 else 'negative'
        records.append({
            'Serie Agro': agro,
            'Variable Macro': macro,
            'Signo del Primer Choque': sign
        })

        # 5) Graficar la IRF individual
        # t = np.arange(len(resp))
        # plt.figure(figsize=(8, 4))
        # plt.plot(t, resp, label=f'IRF {macro}→{agro}')
        # plt.fill_between(t, lower, upper, color='gray', alpha=0.3)
        # plt.plot(t, lower, 'r--', linewidth=1)
        # plt.plot(t, upper, 'r--', linewidth=1)
        # plt.axhline(0, color='k', lw=0.8)
        # plt.title(f'IRF significativa: choque en {macro} → {agro} (p={p})')
        # plt.xlabel('Horizonte')
        # plt.ylabel('Impacto')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

    # 6) Devolver DataFrame con todos los pares significativos
    df_irf = pd.DataFrame(records)
    if df_irf.empty:
        print("No se encontraron IRFs significativas al nivel especificado.")
    return df_irf

df_IFR = plot_macro_to_agro_irf_grid(
    dict_agro=dict_productos_est2,
    dict_macro=dict_macro_est2,
    lag_selection=lags,
    steps=24,
    orth=True,
    alpha=0.05
)

# %%
# 1) Pruebas de estacionariedad ADF para cada serie individual

dict_productos_est,df_productos_est = ajuste_estacionareidad(dict_productos)
# dict_productos_est,df_productos_est = ajuste_estacionareidad(dict_productos_filt)
dict_macro_est,df_macro_est = ajuste_estacionareidad(dict_macro)

display(f'Quedaron {len(dict_productos_est)} productos y {len(dict_macro_est)} variables macro')

# %%
# 2) Eliminar estacionalidad
dict_productos_est2,df_productos_est2 = adjust_seasonality(dict_productos_est)
dict_macro_est2,df_macro_est2 = adjust_seasonality(dict_macro_est)

display(f'Quedaron {len(dict_productos_est2)} productos {len(dict_macro_est2)} variables macro')


# %%
# 3) Selección de lags por AIC/BIC

lags,df_lags = select_var_lags(dict_productos_est2, dict_macro_est2, maxlags=12)

for (agro, macro), info in lags.items():
    print(f"{agro} vs {macro} → AIC: {info['aic']}, BIC: {info['bic']}")

# %%
def plot_macro_to_agro_irf_grid(
    dict_agro, dict_macro, lag_selection, steps=12, orth=True, alpha=0.05
):
    """
    Para cada par (agro, macro) calcula la IRF macro→agro, 
    construye su banda de confianza y, si es significativa en
    algún horizonte, la dibuja en su propia figura.
    """
    z = norm.ppf(1 - alpha/2)
    for (agro, macro), params in lag_selection.items():
        p = params.get('aic', params.get('bic'))
        # 1) armar el DataFrame
        s_ag = dict_agro[agro][agro].dropna()
        s_ma = dict_macro[macro][macro].dropna()
        df = pd.concat([s_ag, s_ma], axis=1, join='inner').dropna()
        df.columns = ['y_ag', 'y_ma']
        if len(df) < p+2:
            continue

        # 2) ajustar VAR y extraer IRF
        try:
            model = VAR(df).fit(p)
            irf_res = model.irf(steps)
        except Exception:
            continue

        irfs = irf_res.irfs       # shape (steps+1, 2, 2)
        covs = irf_res.cov(orth)  # shape (steps+1, 4, 4)

        # 3) tomar solo la respuesta de y_ag al choque en y_ma
        resp = irfs[:, 0, 1]
        errs = np.array([np.sqrt(covs[h, 0*2+1, 0*2+1]) for h in range(len(resp))])

        lower = resp - z * errs
        upper = resp + z * errs

        # 4) chequear si en el horizonte 1 el intervalo no cruza cero
        #    (recordar que `resp`, `lower` y `upper` tienen índices 0..steps)
        h = 1
        sig1 = ((lower[h] > 0) or (upper[h] < 0)) and (not np.isnan(lower[h]))
        if not sig1:
            continue


        # 5) PLOT individual
        t = np.arange(len(resp))
        plt.figure(figsize=(8, 4))
        plt.plot(t, resp, label=f'IRF {macro}→{agro}')
        plt.fill_between(t, lower, upper, color='gray', alpha=0.3)
        plt.plot(t, lower, 'r--', linewidth=1)
        plt.plot(t, upper, 'r--', linewidth=1)
        plt.axhline(0, color='k', lw=0.8)
        plt.title(f'IRF significativa: choque en {macro} → {agro} (p={p})')
        plt.xlabel('Horizonte')
        plt.ylabel('Impacto')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Graficas IFR
irfs_dict = plot_macro_to_agro_irf_grid(
    dict_agro=dict_productos_est2,
    dict_macro=dict_macro_est2,
    lag_selection=lags,
    steps=16,
    orth=True,
    alpha=0.05   # Intervalos del 95%
)

# %%
# Resumen general

resumen = (
    df_IFR
      .groupby('Variable Macro')['Signo del Primer Choque']
      .value_counts()
      .unstack(fill_value=0)               # columnas 'negative' y 'positive'
      .rename(columns={'negative':'count_negative',
                       'positive':'count_positive'})
)

# Añadimos la columna de total de apariciones
resumen['total'] = resumen['count_positive'] + resumen['count_negative']

# Reordenamos las columnas
resumen = resumen[['total', 'count_positive', 'count_negative']]

# Reiniciamos índice para que 'macro' sea columna
resumen = resumen.reset_index().rename(columns={'macro':'Variable Macro',
                                                'total':'Total Apariciones',
                                                'count_positive':'Veces Positive',
                                                'count_negative':'Veces Negative'})


# %% [markdown]
# # 3 modelos ML

# %%
import itertools
from tensorflow.keras.layers import Input  # ← para modelos
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import ReduceLROnPlateau



# ── Funciones comunes ─────────────────────────────────────────────────────────
def create_supervised(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size])
    return np.array(X, dtype='float32')[..., np.newaxis], np.array(y, dtype='float32')

# ── Clase ELM ─────────────────────────────────────────────────────────────────
class ELM:
    def __init__(self, n_hidden, activation='tanh'):
        self.n_hidden  = n_hidden
        self.activation = np.tanh if activation=='tanh' else (lambda x: np.maximum(0, x))
    def fit(self, X, y):
        # X: (n_samples, window_size, 1) → aplanar
        X2 = X.reshape(X.shape[0], X.shape[1])
        self.W = np.random.uniform(-1,1,(X2.shape[1], self.n_hidden))
        self.b = np.random.uniform(-1,1,(self.n_hidden,))
        H = self.activation(X2.dot(self.W) + self.b)
        self.beta = pinv(H).dot(y)
    def predict(self, X):
        X2 = X.reshape(X.shape[0], X.shape[1])
        H = self.activation(X2.dot(self.W) + self.b)
        return H.dot(self.beta)

# ── GA-ELM fitness ─────────────────────────────────────────────────────────────
def fitness_elm(ga, sol, idx, serie_norm, train_frac):
    w, h = int(sol[0]), int(sol[1])
    if w < 1 or h < 1 or len(serie_norm) < w + 10:
        return -1e6
    X, y = create_supervised(serie_norm, w)
    split = int(len(X) * train_frac)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    elm = ELM(n_hidden=h)
    elm.fit(X_tr, y_tr)
    y_pred = elm.predict(X_va)
    rmse = np.sqrt(mean_squared_error(y_va, y_pred))
    return -rmse

def optimize_elm_for_series(serie, train_frac, generations=20, pop_size=30):
    raw = serie.dropna().values.astype('float32').reshape(-1,1)
    # z-score sobre train
    ss = StandardScaler()
    cutoff = int(len(raw)*train_frac)
    ss.fit(raw[:cutoff])
    norm = ss.transform(raw).flatten()

    gene_space = [
        {'low':1,'high':24,'step':1},   # window_size
        {'low':5,'high':200,'step':1},  # n_hidden
    ]
    ga = pygad.GA(
        num_generations=generations,
        sol_per_pop=pop_size,
        num_parents_mating=pop_size//2,
        num_genes=2,
        gene_space=gene_space,
        fitness_func=lambda ga, s, i: fitness_elm(ga, s, i, norm, train_frac),
        parent_selection_type="tournament",
        keep_parents=5,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=20,
        random_seed=42
    )
    ga.run()
    best, _, _ = ga.best_solution()
    w, h = int(best[0]), int(best[1])

    # evaluar en test
    X, y = create_supervised(norm, w)
    split = int(len(X)*train_frac)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]
    elm = ELM(n_hidden=h)
    elm.fit(X_tr, y_tr)
    y_pred = elm.predict(X_te)
    return {
        'window_size': w,
        'n_hidden':    h,
        'rmse':        np.sqrt(mean_squared_error(y_te, y_pred)),
        'mae':         mean_absolute_error(y_te, y_pred),
        'mape (%)':    np.mean(np.abs((y_te - y_pred)/y_te))*100
    }

# ── Modelos DL ────────────────────────────────────────────────────────────────
def build_elman(window_size, units=32, learning_rate=1e-3):
    inp = Input((window_size,1))
    x   = SimpleRNN(units, activation='tanh')(inp)
    out = Dense(1)(x)
    m   = Model(inp,out); m.compile(RMSprop(learning_rate),'mse'); return m

def build_lstnet(window_size, conv_filters=64, gru_units=50, learning_rate=1e-3):
    inp = Input((window_size,1))
    x   = Conv1D(conv_filters,3,activation='relu',padding='causal')(inp)
    x   = GRU(gru_units)(x)
    out = Dense(1)(x)
    m   = Model(inp,out); m.compile(Adam(learning_rate),'mse'); return m

model_configs = {
    'Elman': {'builder': build_elman,  'param_grid': {'window_size':[6,12,24],'units':[16,32],'learning_rate':[1e-3]}},
    'LSTNet':{'builder': build_lstnet, 'param_grid': {'window_size':[6,12,24],'conv_filters':[32,64],'gru_units':[32,50],'learning_rate':[1e-3]}}
}

# ── Torneo de modelos ─────────────────────────────────────────────────────────
def evaluate_tournament(series_dict, model_configs, train_frac=0.8, val_split=0.1, epochs=30, batch_size=32):
    records = []
    for name, serie in series_dict.items():
        raw = serie.dropna().values.reshape(-1,1)
        # MinMax
        mm = MinMaxScaler(); cutoff=int(len(raw)*train_frac); mm.fit(raw[:cutoff])
        norm_mm = mm.transform(raw).flatten()

        # DL: Elman & LSTNet
        for model_name,cfg in model_configs.items():
            for combo in itertools.product(*cfg['param_grid'].values()):
                params = dict(zip(cfg['param_grid'].keys(), combo))
                w = params['window_size']
                X,y = create_supervised(norm_mm,w)
                if len(X)<10: continue
                split=int(len(X)*train_frac)
                X_tr,X_te = X[:split],X[split:]; y_tr,y_te = y[:split],y[split:]
                m = cfg['builder'](**params)
                m.fit(X_tr,y_tr,validation_split=val_split,epochs=epochs,batch_size=batch_size,callbacks=[EarlyStopping('val_loss',patience=5,restore_best_weights=True)],verbose=0)
                y_pred=m.predict(X_te).flatten()
                records.append({
                    'serie':name,'modelo':model_name,**params,
                    'rmse':np.sqrt(mean_squared_error(y_te,y_pred)),
                    'mae': mean_absolute_error(y_te,y_pred),
                    'mape (%)': np.mean(np.abs((y_te-y_pred)/y_te))*100
                })

        # GA-ELM
        elm_res = optimize_elm_for_series(serie, train_frac)
        elm_res.update({'serie':name,'modelo':'GA-ELM'})
        records.append(elm_res)

    return pd.DataFrame(records)

# ── Selección del mejor y proyección ────────────────────────────────────────────
def select_and_forecast(series_dict, df_results):
    """
    1) Del df_results elige por serie el modelo de menor RMSE.
    2) Reentrena ese modelo con todos los datos y genera 12 meses de proyección.
    Devuelve:
      - best: dict con la configuración elegida por serie
      - forecasts: dict con pd.Series (original + 12 pronósticos)
    """
    best = {}
    forecasts = {}

    for serie_name, grp in df_results.groupby('serie'):
        # 1) seleccionar la fila con RMSE mínimo
        row = grp.loc[grp['rmse'].idxmin()].to_dict()
        best[serie_name] = row

        # preparar datos
        orig_series = series_dict[serie_name].dropna()
        raw = orig_series.values.astype('float32').reshape(-1,1)
        last_date = orig_series.index[-1]
        future_index = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(),
            periods=12, freq='M'
        )

        model_type = row['modelo']
        w = int(row['window_size'])

        if model_type == 'GA-ELM':
            # Z-score + ELM
            ss = StandardScaler().fit(raw)
            norm = ss.transform(raw).flatten()
            X, y = create_supervised(norm, w)
            elm = ELM(n_hidden=int(row['n_hidden']))
            elm.fit(X, y)
            window = norm[-w:].tolist()
            preds = []
            for _ in range(12):
                x_in = np.array(window[-w:])[None, ...]
                p = elm.predict(x_in)[0]
                window.append(p)
                preds.append(p)
            preds = ss.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

        else:
            # MinMax + Red Keras
            mm = MinMaxScaler().fit(raw)
            norm = mm.transform(raw).flatten()

            # extraer e castear hiperparámetros correctamente
            params = {}
            for k in model_configs[model_type]['param_grid']:
                if k == 'learning_rate':
                    params[k] = float(row[k])
                else:
                    params[k] = int(row[k])

            X, y = create_supervised(norm, w)
            # reconstruir y entrenar
            builder = model_configs[model_type]['builder']
            model = builder(**params)
            model.fit(X, y,
                      epochs=50, batch_size=32,
                      callbacks=[EarlyStopping('loss', patience=5, restore_best_weights=True)],
                      verbose=0)

            window = norm[-w:].tolist()
            preds = []
            for _ in range(12):
                x_in = np.array(window[-w:])[None, ..., None]
                p = model.predict(x_in).flatten()[0]
                window.append(p)
                preds.append(p)
            preds = mm.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

        # concatenar serie original con pronóstico
        forecasts[serie_name] = pd.concat([
            orig_series,
            pd.Series(preds, index=future_index)
        ])

    return best, forecasts


df_comp = evaluate_tournament(dict_productos_filt, model_configs)
best_models, dict_with_forecasts = select_and_forecast(dict_productos_filt, df_comp)

# %%
df_mejor_modelo = df_comp.loc[df_comp.groupby('serie')['rmse'].idxmin()].reset_index(drop=True)

# Crear columna de hiperparámetros
def extraer_hiperparametros(row):
    modelo = row['modelo']
    if modelo == 'GA-ELM':
        return {'window_size': row['window_size'], 'n_hidden': row['n_hidden']}
    elif modelo == 'Elman':
        return {'window_size': row['window_size'], 'units': row['units'], 'learning_rate': row['learning_rate']}
    elif modelo == 'LSTNet':
        return {'window_size': row['window_size'], 'conv_filters': row['conv_filters'], 'gru_units': row['gru_units'], 'learning_rate': row['learning_rate']}
    else:
        return {}

df_mejor_modelo['hiperparametros'] = df_mejor_modelo.apply(extraer_hiperparametros, axis=1)

# Si quieres solo las columnas principales + hiperparámetros
df_mejor_modelo = (
    df_mejor_modelo
    [['serie','modelo','rmse','mae','mape (%)','hiperparametros']]
    .round({'rmse': 2, 'mae': 2, 'mape (%)': 2})
)


df_to_image_optimized(
    df_mejor_modelo, 
    'informe_variables.png',
    font_size=11,
    max_col_width=200  # Ajustar según necesidad
)

# %%
df_mejor_modelo1 = df_mejor_modelo[['serie','modelo','rmse','mae','mape (%)']]
df_to_image_optimized(
    df_mejor_modelo1, 
    'informe_variables.png',
    font_size=11,
    max_col_width=200  # Ajustar según necesidad
    )

# %%
df_mejor_modelo1.to_excel(
    'informe_ML.xlsx',)

# %%
df_mejor_modelo

# %%



