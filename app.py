
import streamlit as st
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Saber 11 | Ciencia de Datos",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# ESTILOS
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #F8F9FC;
    }

    h1, h2, h3 {
        color: #243B6B;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #E3E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CARGA DE DATOS
# ==========================================

@st.cache_data
def cargar_datos():
    ruta = Path("data/saber11_modelado.csv")
    return pd.read_csv(ruta)

try:
    df = cargar_datos()
except Exception as e:
    st.error(f"No se pudo cargar el dataset: {e}")
    st.stop()

# ==========================================
# MENÚ
# ==========================================

st.sidebar.title("🎓 Saber 11")
st.sidebar.caption("Proyecto final - Ciencia de Datos")

pagina = st.sidebar.radio(
    "Navegación",
    ["Inicio", "Exploración", "Predicción", "Resultados"]
)

st.sidebar.divider()
st.sidebar.info("Fuente: datos.gov.co | ICFES")

# ==========================================
# INICIO
# ==========================================

if pagina == "Inicio":
    st.title("🎓 Análisis del desempeño académico")
    st.write(
        "Proyecto de ciencia de datos basado en los "
        "resultados de las pruebas Saber 11 en Colombia."
    )

    st.subheader("¿Qué buscamos?")
    st.write(
        "Analizar la relación entre factores socioeconómicos, "
        "familiares y educativos y el desempeño académico, "
        "mediante técnicas de Machine Learning."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Estudiantes", f"{len(df):,}")

    with col2:
        st.metric("Variables", len(df.columns))

    with col3:
        st.metric(
            "Puntaje promedio",
            f"{df['punt_global'].mean():.1f}"
        )

    st.subheader("Fuente de datos")
    st.link_button(
        "Consultar dataset oficial",
        "https://www.datos.gov.co/Education/Saber-11-2020-2/rnvb-vnyh"
    )

# ==========================================
# EXPLORACIÓN
# ==========================================

elif pagina == "Exploración":
    st.title("📊 Exploración de datos")

    st.write(
        "Visualización de la distribución de puntajes "
        "y su relación con algunas características."
    )

    st.subheader("Distribución del puntaje global")
    st.bar_chart(
        df["punt_global"]
        .value_counts(bins=20)
        .sort_index()
    )

    st.subheader("Puntaje promedio por estrato")
    promedio_estrato = (
        df.groupby("fami_estratovivienda")["punt_global"]
        .mean()
    )
    st.bar_chart(promedio_estrato)

    st.subheader("Puntaje promedio según acceso a internet")
    promedio_internet = (
        df.groupby("fami_tieneinternet")["punt_global"]
        .mean()
    )
    st.bar_chart(promedio_internet)

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(20), use_container_width=True)

# ==========================================
# PREDICCIÓN
# ==========================================

elif pagina == "Predicción":
    st.title("🔮 Predicción del desempeño académico")

    st.write(
        "Esta sección permitirá ingresar características "
        "de un estudiante y obtener una predicción."
    )

    st.warning(
        "El modelo de Machine Learning todavía está "
        "en desarrollo. La predicción estará disponible "
        "cuando se integre el modelo seleccionado."
    )

    st.subheader("Variables del estudiante")

    variables = [
        "fami_estratovivienda",
        "fami_educacionmadre",
        "fami_educacionpadre",
        "fami_tieneinternet",
        "fami_tienecomputador",
        "cole_naturaleza",
        "cole_area_ubicacion",
        "estu_genero"
    ]

    with st.form("formulario_prediccion"):
        for variable in variables:
            opciones = (
                df[variable]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            st.selectbox(
                variable.replace("_", " ").title(),
                sorted(opciones),
                key=variable
            )

        enviar = st.form_submit_button("Realizar predicción")

    if enviar:
        st.info(
            "Formulario registrado en esta sesión. "
            "La predicción se habilitará al integrar "
            "el modelo entrenado."
        )

# ==========================================
# RESULTADOS
# ==========================================

elif pagina == "Resultados":
    st.title("🏆 Comparación de modelos")

    st.write(
        "En esta sección presentaremos las métricas "
        "de los tres modelos de Machine Learning."
    )

    st.info(
        "Pendiente de integrar los resultados de "
        "entrenamiento y evaluación."
    )

    st.subheader("Criterios de evaluación")
    st.write(
        "Se compararán las métricas apropiadas para "
        "el problema definido, como Accuracy, "
        "Precision, Recall y F1-score si se utiliza clasificación."
    )

    st.subheader("Modelo seleccionado")
    st.write(
        "Se documentará la elección del modelo final "
        "a partir de sus resultados y su utilidad."
    )

# ==========================================
# PIE DE PÁGINA
# ==========================================

st.divider()
st.caption(
    "Universidad de San Buenaventura | "
    "Proyecto Final de Ciencia de Datos | 2026"
)
