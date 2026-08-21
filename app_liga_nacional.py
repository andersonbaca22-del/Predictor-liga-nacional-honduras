"""
Predictor Liga Nacional de Honduras - Modelo Poisson + Dixon-Coles
--------------------------------------------------------------
Interfaz visual con Streamlit. No requiere tocar código para usarla:
- Seleccionar equipos y ver la predicción de un partido
- Agregar un partido jugado nuevo (se guarda en el Excel)
- Agregar un equipo nuevo a la lista

Para ejecutar (una vez instalado Python y las librerías):
    pip install streamlit pandas numpy statsmodels scipy openpyxl
    streamlit run app_liga_nacional.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson

# ----------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------
ARCHIVO_EXCEL = "experimento.xlsx"   # cambiá esto por el nombre real de tu archivo
HOJA_PARTIDOS = "Partidos"           # nombre de la hoja con los partidos
MAX_GOLES = 10
RHO_DIXON_COLES = -0.1
XI_PONDERADO = 0.0008                # el que calibraste

st.set_page_config(page_title="Predictor Liga Nacional", layout="centered")
st.title("⚽ Predictor Liga Nacional de Honduras")
st.caption("Modelo Poisson GLM + corrección Dixon-Coles + ponderado temporal")


# ----------------------------------------------------------------
# CARGA DE DATOS (con cache para no releer el excel en cada click)
# ----------------------------------------------------------------
@st.cache_data
def cargar_datos(archivo, hoja):
    partidos = pd.read_excel(archivo, sheet_name=hoja)
    partidos['fecha'] = pd.to_datetime(partidos['fecha'])
    return partidos


def construir_datos_largos(partidos):
    local_df = partidos[['fecha', 'local', 'visitante', 'goles_local']].copy()
    local_df.columns = ['fecha', 'equipo', 'rival', 'goles']
    local_df['es_local'] = 1

    visitante_df = partidos[['fecha', 'visitante', 'local', 'goles_visitante']].copy()
    visitante_df.columns = ['fecha', 'equipo', 'rival', 'goles']
    visitante_df['es_local'] = 0

    datos_largos = pd.concat([local_df, visitante_df], ignore_index=True)

    fecha_prediccion = pd.Timestamp.today()
    datos_largos['dias'] = (fecha_prediccion - datos_largos['fecha']).dt.days
    datos_largos['peso'] = np.exp(-XI_PONDERADO * datos_largos['dias'])

    return datos_largos


@st.cache_resource
def entrenar_modelo(datos_largos):
    modelo = smf.glm(
        formula='goles ~ equipo + rival + es_local',
        data=datos_largos,
        family=sm.families.Poisson(),
        freq_weights=datos_largos['peso']
    ).fit()
    return modelo


K_ENCOGIMIENTO = 10  # cuanto más alto, más se encoge un equipo con pocos partidos hacia el promedio de liga


def calcular_lambda(modelo, equipo_local, equipo_visitante, datos_largos, k=K_ENCOGIMIENTO):
    """
    Calcula lambda para ambos equipos. Los coeficientes de equipos con pocos
    partidos jugados se "encogen" hacia 0 (el promedio de la liga), para no
    confiar demasiado en una muestra chica (ej: equipos recién ascendidos).
    """
    conteo_partidos = datos_largos['equipo'].value_counts()

    def coef_encogido(nombre_param, equipo):
        if nombre_param in modelo.params:
            n = conteo_partidos.get(equipo, 0)
            factor = n / (n + k)
            return modelo.params[nombre_param] * factor
        return 0

    log_lambda_local = modelo.params['Intercept']
    log_lambda_local += coef_encogido(f'equipo[T.{equipo_local}]', equipo_local)
    log_lambda_local += coef_encogido(f'rival[T.{equipo_visitante}]', equipo_visitante)
    log_lambda_local += modelo.params['es_local']
    lambda_local = np.exp(log_lambda_local)

    log_lambda_visitante = modelo.params['Intercept']
    log_lambda_visitante += coef_encogido(f'equipo[T.{equipo_visitante}]', equipo_visitante)
    log_lambda_visitante += coef_encogido(f'rival[T.{equipo_local}]', equipo_local)
    lambda_visitante = np.exp(log_lambda_visitante)

    return lambda_local, lambda_visitante


def tau_dixon_coles(x, y, lambda_local, lambda_visitante, rho):
    if x == 0 and y == 0:
        return 1 - (lambda_local * lambda_visitante * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_local * rho)
    elif x == 1 and y == 0:
        return 1 + (lambda_visitante * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def calcular_matriz_y_probabilidades(lambda_local, lambda_visitante):
    prob_local = [poisson.pmf(i, lambda_local) for i in range(MAX_GOLES + 1)]
    prob_visitante = [poisson.pmf(i, lambda_visitante) for i in range(MAX_GOLES + 1)]
    matriz = np.outer(prob_local, prob_visitante)

    matriz_corregida = matriz.copy()
    for i in range(2):
        for j in range(2):
            factor = tau_dixon_coles(i, j, lambda_local, lambda_visitante, RHO_DIXON_COLES)
            matriz_corregida[i][j] = matriz[i][j] * factor

    p_local = sum(matriz_corregida[i][j] for i in range(MAX_GOLES + 1) for j in range(MAX_GOLES + 1) if i > j)
    p_empate = sum(matriz_corregida[i][j] for i in range(MAX_GOLES + 1) for j in range(MAX_GOLES + 1) if i == j)
    p_visitante = sum(matriz_corregida[i][j] for i in range(MAX_GOLES + 1) for j in range(MAX_GOLES + 1) if i < j)

    suma = p_local + p_empate + p_visitante
    p_local, p_empate, p_visitante = p_local / suma, p_empate / suma, p_visitante / suma

    return matriz_corregida, p_local, p_empate, p_visitante


# ----------------------------------------------------------------
# CARGA INICIAL
# ----------------------------------------------------------------
try:
    partidos = cargar_datos(ARCHIVO_EXCEL, HOJA_PARTIDOS)
except FileNotFoundError:
    st.error(f"No encontré el archivo '{ARCHIVO_EXCEL}'. Colocalo en la misma carpeta que este script.")
    st.stop()

datos_largos = construir_datos_largos(partidos)
modelo = entrenar_modelo(datos_largos)

equipos_conocidos = sorted(set(partidos['local'].unique()) | set(partidos['visitante'].unique()))

# ----------------------------------------------------------------
# PESTAÑAS DE LA APP
# ----------------------------------------------------------------
tab_predecir, tab_agregar_partido, tab_agregar_equipo = st.tabs(
    ["🔮 Predecir partido", "➕ Agregar partido", "🆕 Agregar equipo"]
)

# --- TAB 1: PREDECIR ---
with tab_predecir:
    col1, col2 = st.columns(2)
    with col1:
        equipo_local = st.selectbox("Equipo local", equipos_conocidos, key="local_sel")
    with col2:
        equipo_visitante = st.selectbox(
            "Equipo visitante",
            [e for e in equipos_conocidos if e != equipo_local],
            key="visit_sel"
        )

    if st.button("Calcular predicción", type="primary"):
        conteo_partidos = datos_largos['equipo'].value_counts()
        n_local = conteo_partidos.get(equipo_local, 0)
        n_visitante = conteo_partidos.get(equipo_visitante, 0)
        UMBRAL_POCOS_PARTIDOS = 15

        if n_local < UMBRAL_POCOS_PARTIDOS or n_visitante < UMBRAL_POCOS_PARTIDOS:
            equipos_con_poca_muestra = [
                f"{eq} ({n} partidos)"
                for eq, n in [(equipo_local, n_local), (equipo_visitante, n_visitante)]
                if n < UMBRAL_POCOS_PARTIDOS
            ]
            st.warning(
                "⚠️ Predicción menos confiable: " + ", ".join(equipos_con_poca_muestra) +
                " tiene(n) pocos partidos registrados. El modelo ajustó sus coeficientes "
                "hacia el promedio de la liga para compensar, pero conviene tomar este "
                "resultado con más cautela hasta que acumulen más historial."
            )

        lambda_local, lambda_visitante = calcular_lambda(modelo, equipo_local, equipo_visitante, datos_largos)
        matriz, p_local, p_empate, p_visitante = calcular_matriz_y_probabilidades(lambda_local, lambda_visitante)

        st.subheader("Goles esperados (λ)")
        c1, c2 = st.columns(2)
        c1.metric(equipo_local, f"{lambda_local:.2f}")
        c2.metric(equipo_visitante, f"{lambda_visitante:.2f}")

        st.subheader("Probabilidad de resultado")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Gana {equipo_local}", f"{p_local*100:.1f}%")
        c2.metric("Empate", f"{p_empate*100:.1f}%")
        c3.metric(f"Gana {equipo_visitante}", f"{p_visitante*100:.1f}%")

        st.subheader("Marcadores más probables")
        resultados = []
        for i in range(MAX_GOLES + 1):
            for j in range(MAX_GOLES + 1):
                resultados.append((i, j, matriz[i][j]))
        resultados.sort(key=lambda x: x[2], reverse=True)

        tabla_marcadores = pd.DataFrame(
            [(f"{i} - {j}", f"{p*100:.2f}%") for i, j, p in resultados[:8]],
            columns=["Marcador", "Probabilidad"]
        )
        st.table(tabla_marcadores)

# --- TAB 2: AGREGAR PARTIDO ---
with tab_agregar_partido:
    st.write("Cargá un partido ya jugado. Se guarda directo en el Excel.")

    with st.form("form_partido"):
        c1, c2 = st.columns(2)
        fecha_partido = c1.date_input("Fecha")
        temporada_partido = c2.text_input("Temporada (ej: Apertura 2026)")

        c1, c2 = st.columns(2)
        local_nuevo = c1.selectbox("Local", equipos_conocidos, key="local_form")
        visitante_nuevo = c2.selectbox(
            "Visitante", [e for e in equipos_conocidos if e != local_nuevo], key="visit_form"
        )

        c1, c2 = st.columns(2)
        goles_local_nuevo = c1.number_input("Goles local", min_value=0, step=1)
        goles_visitante_nuevo = c2.number_input("Goles visitante", min_value=0, step=1)

        c1, c2 = st.columns(2)
        jornada_nueva = c1.text_input("Jornada (vacío si es liguilla)")
        es_liguilla_nueva = c2.selectbox("¿Es liguilla?", [0, 1])

        ronda_nueva = st.text_input("Ronda (vacío si es fase regular)")

        enviado = st.form_submit_button("Guardar partido")

        if enviado:
            nueva_fila = pd.DataFrame([{
                "fecha": fecha_partido,
                "temporada": temporada_partido,
                "jornada": jornada_nueva,
                "local": local_nuevo,
                "visitante": visitante_nuevo,
                "goles_local": goles_local_nuevo,
                "goles_visitante": goles_visitante_nuevo,
                "es_liguilla": es_liguilla_nueva,
                "ronda": ronda_nueva,
            }])
            partidos_actualizado = pd.concat([partidos, nueva_fila], ignore_index=True)
            partidos_actualizado.to_excel(ARCHIVO_EXCEL, sheet_name=HOJA_PARTIDOS, index=False)
            st.success("Partido guardado. Recargá la página para que el modelo lo incluya en el próximo cálculo.")
            st.cache_data.clear()
            st.cache_resource.clear()

# --- TAB 3: AGREGAR EQUIPO ---
with tab_agregar_equipo:
    st.write("Agregá un equipo nuevo a la lista disponible (por ejemplo, un ascendido).")
    st.info(
        "Esto solo lo agrega a la lista de selección. Para que el modelo tenga un "
        "coeficiente confiable de ese equipo, necesita que cargues varios partidos "
        "jugados por él en la pestaña 'Agregar partido'."
    )
    nuevo_equipo = st.text_input("Nombre del equipo nuevo")
    if st.button("Agregar a la lista"):
        if nuevo_equipo and nuevo_equipo not in equipos_conocidos:
            st.session_state.setdefault("equipos_extra", [])
            st.session_state["equipos_extra"].append(nuevo_equipo)
            st.success(
                f"'{nuevo_equipo}' anotado. Para que quede guardado de forma permanente, "
                "cargale al menos un partido en la pestaña 'Agregar partido' — "
                "ahí se incorpora solo a la base de datos."
            )
        elif nuevo_equipo in equipos_conocidos:
            st.warning("Ese equipo ya existe en la lista.")
