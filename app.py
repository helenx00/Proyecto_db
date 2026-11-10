import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Saber 11 | Ciencia de Datos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PALETA DE COLORES
# ==========================================

AZUL = "#6C9FE8"
AZUL_OSCURO = "#243B6B"
ROSADO = "#EBAFC5"
BLANCO = "#FFFFFF"
FONDO = "#F8F9FC"

# ==========================================
# 3. ESTILOS
# ==========================================

st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background-color: #F8F9FC;
        color: #243B6B;
    }

    /* Títulos */
    h1, h2, h3 {
        color: #243B6B !important;
    }

    /* Textos del contenido principal */
    .stMain p,
    .stMain li,
    .stMain label {
        color: #344563;
    }

    /* Tarjetas de métricas */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #DCE4EF;
    }

    div[data-testid="stMetricValue"] {
        color: #243B6B;
    }

    div[data-testid="stMetricLabel"] {
        color: #526581;
    }

    /* Menú lateral */
    section[data-testid="stSidebar"] {
        background-color: #18263D;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2 {
        color: #FFFFFF !important;
    }

    /* Botón del dataset */
    div.stLinkButton a {
        background-color: #243B6B !important;
        border-radius: 10px !important;
        border: none !important;
    }

    div.stLinkButton a p {
        color: #FFFFFF !important;
    }

    /* Botón de predicción */
    div.stFormSubmitButton button {
        background-color: #243B6B !important;
        border: none !important;
        border-radius: 10px !important;
    }

    div.stFormSubmitButton button p {
        color: #FFFFFF !important;
    }

    /* Formulario */
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE4EF !important;
        border-radius: 14px !important;
        padding: 20px !important;
    }

    /* CASILLAS BLANCAS */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE4EF !important;
        border-radius: 10px !important;
        color: #243B6B !important;
    }

    div[data-baseweb="select"] span {
        color: #243B6B !important;
    }

    div[data-baseweb="select"] svg {
        color: #243B6B !important;
        fill: #243B6B !important;
    }

    /* Opciones desplegables */
    ul[role="listbox"] {
        background-color: #FFFFFF !important;
    }

    li[role="option"] {
        color: #243B6B !important;
    }

    li[role="option"]:hover {
        background-color: #E8F0FC !important;
    }

    /* Tabla */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. RECUADROS PERSONALIZADOS
# ==========================================

def mostrar_aviso(mensaje, titulo="Información"):
    st.markdown(
        f"""
        <div style="
            background-color:#E8F0FC;
            color:#243B6B;
            padding:16px 20px;
            border-radius:12px;
            border:1px solid #C8D9F3;
            margin:15px 0 25px 0;
            line-height:1.6;
        ">
            <strong>{titulo}:</strong> {mensaje}
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 5. ESTILO DE LAS GRÁFICAS
# ==========================================

def preparar_grafica(ax, fig, titulo, eje_x, eje_y):
    fig.patch.set_facecolor(BLANCO)
    ax.set_facecolor(BLANCO)

    ax.set_title(
        titulo,
        fontsize=15,
        color=AZUL_OSCURO,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel(eje_x, fontsize=11)
    ax.set_ylabel(eje_y, fontsize=11)

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", colors="#344563")

# ==========================================
# 6. CARGAR DATOS
# ==========================================

@st.cache_data
def cargar_datos():
    ruta = (
        Path(__file__).resolve().parent
        / "data"
        / "saber11_modelado.csv"
    )

    datos = pd.read_csv(ruta)

    datos["punt_global"] = pd.to_numeric(
        datos["punt_global"],
        errors="coerce"
    )

    return datos

try:
    df = cargar_datos()
except Exception as e:
    st.error(f"No se pudo cargar el dataset: {e}")
    st.stop()

# ==========================================
# 7. MENÚ LATERAL
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
# 8. INICIO
# ==========================================

if pagina == "Inicio":

    st.title("🎓 Análisis del desempeño académico")

    st.write(
        "Proyecto de ciencia de datos basado en los "
        "resultados de las pruebas Saber 11 en Colombia."
    )

    st.subheader("¿Qué buscamos?")

    st.write(
        "Analizar la relación entre factores "
        "socioeconómicos, familiares y educativos "
        "y el desempeño académico mediante "
        "técnicas de Machine Learning."
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

    st.write(
        "Se utiliza una muestra de 10.000 registros "
        "del dataset oficial Saber 11 disponible "
        "en el Portal de Datos Abiertos de Colombia."
    )

    st.link_button(
        "Consultar dataset oficial",
        "https://www.datos.gov.co/Education/Saber-11-2020-2/rnvb-vnyh"
    )

    mostrar_aviso(
        "Los datos corresponden a los primeros "
        "10.000 registros consultados mediante la API. "
        "No constituyen una muestra aleatoria "
        "representativa de todos los estudiantes.",
        "Nota sobre la muestra"
    )

# ==========================================
# 9. EXPLORACIÓN
# ==========================================

elif pagina == "Exploración":

    st.title("📊 Exploración de datos")

    st.write(
        "Visualización de la distribución de puntajes "
        "y su relación con características "
        "socioeconómicas y educativas."
    )

    # --------------------------------------
    # GRÁFICA 1: DISTRIBUCIÓN
    # --------------------------------------

    st.subheader("Distribución del puntaje global")

    fig1, ax1 = plt.subplots(figsize=(11, 4.5))

    ax1.hist(
        df["punt_global"].dropna(),
        bins=20,
        color=AZUL,
        edgecolor="white",
        alpha=0.9
    )

    preparar_grafica(
        ax1,
        fig1,
        "Distribución de los puntajes Saber 11",
        "Puntaje global",
        "Cantidad de estudiantes"
    )

    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.caption(
        "El histograma muestra la cantidad de "
        "estudiantes por intervalo de puntaje global."
    )

    # --------------------------------------
    # GRÁFICA 2: ESTRATO
    # --------------------------------------

    st.subheader("Puntaje promedio por estrato")

    promedio_estrato = (
        df.groupby("fami_estratovivienda")["punt_global"]
        .mean()
    )

    orden_estratos = [
        "Sin Estrato",
        "Estrato 1",
        "Estrato 2",
        "Estrato 3",
        "Estrato 4",
        "Estrato 5",
        "Estrato 6"
    ]

    promedio_estrato = (
        promedio_estrato
        .reindex(orden_estratos)
        .dropna()
    )

    fig2, ax2 = plt.subplots(figsize=(11, 4.5))

    barras2 = ax2.bar(
        promedio_estrato.index,
        promedio_estrato.values,
        color=AZUL,
        edgecolor="white",
        width=0.65
    )

    preparar_grafica(
        ax2,
        fig2,
        "Puntaje promedio según estrato socioeconómico",
        "Estrato socioeconómico",
        "Puntaje promedio"
    )

    ax2.tick_params(axis="x", rotation=20)

    if not promedio_estrato.empty:
        ax2.set_ylim(0, promedio_estrato.max() * 1.15)

    for barra in barras2:
        altura = barra.get_height()

        ax2.text(
            barra.get_x() + barra.get_width() / 2,
            altura + 2,
            f"{altura:.1f}",
            ha="center",
            fontsize=10,
            color=AZUL_OSCURO
        )

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        "Comparación descriptiva del puntaje "
        "promedio entre los diferentes estratos."
    )

    # --------------------------------------
    # GRÁFICA 3: INTERNET
    # --------------------------------------

    st.subheader(
        "Puntaje promedio según acceso a internet"
    )

    datos_internet = df[
        ["fami_tieneinternet", "punt_global"]
    ].copy()

    datos_internet["fami_tieneinternet"] = (
        datos_internet["fami_tieneinternet"]
        .astype("string")
        .str.strip()
        .replace({
            "Sí": "Si",
            "SI": "Si",
            "NO": "No"
        })
    )

    promedio_internet = (
        datos_internet
        .groupby("fami_tieneinternet")["punt_global"]
        .mean()
        .reindex(["No", "Si"])
        .dropna()
    )

    fig3, ax3 = plt.subplots(figsize=(9, 4.5))

    colores_internet = [
        ROSADO if categoria == "No" else AZUL
        for categoria in promedio_internet.index
    ]

    barras3 = ax3.bar(
        promedio_internet.index,
        promedio_internet.values,
        color=colores_internet,
        edgecolor="white",
        width=0.5
    )

    preparar_grafica(
        ax3,
        fig3,
        "Relación entre acceso a internet y puntaje global",
        "¿Tiene internet en casa?",
        "Puntaje promedio"
    )

    if not promedio_internet.empty:
        ax3.set_ylim(0, promedio_internet.max() * 1.15)

    for barra in barras3:
        altura = barra.get_height()

        ax3.text(
            barra.get_x() + barra.get_width() / 2,
            altura + 2,
            f"{altura:.1f}",
            ha="center",
            fontsize=11,
            color=AZUL_OSCURO
        )

    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "Comparación descriptiva del puntaje "
        "promedio según disponibilidad "
        "de internet en el hogar."
    )

    mostrar_aviso(
        "Las diferencias observadas representan "
        "asociaciones estadísticas y no demuestran "
        "relaciones de causa y efecto.",
        "Importante"
    )

    # --------------------------------------
    # VISTA PREVIA DEL DATASET
    # --------------------------------------

    st.subheader("📋 Vista previa del dataset")

    st.write(
        "A continuación se muestran los primeros "
        "20 registros del conjunto de datos "
        "utilizado para el modelado."
    )

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Dataset: {len(df):,} registros "
        f"y {len(df.columns)} variables."
    )

# ==========================================
# 10. PREDICCIÓN
# ==========================================

elif pagina == "Predicción":

    st.title("🔮 Predicción del desempeño académico")

    st.write(
        "Esta sección permitirá ingresar "
        "características de un estudiante "
        "y obtener una predicción mediante "
        "el modelo seleccionado."
    )

    mostrar_aviso(
        "El modelo de Machine Learning todavía "
        "está en desarrollo. La predicción estará "
        "disponible cuando se integre.",
        "Modelo pendiente"
    )

    st.subheader("Características del estudiante")

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

    etiquetas = {
        "fami_estratovivienda": "Estrato socioeconómico",
        "fami_educacionmadre": "Educación de la madre",
        "fami_educacionpadre": "Educación del padre",
        "fami_tieneinternet": "¿Tiene internet en casa?",
        "fami_tienecomputador": "¿Tiene computador?",
        "cole_naturaleza": "Naturaleza del colegio",
        "cole_area_ubicacion": "Ubicación del colegio",
        "estu_genero": "Género del estudiante"
    }

    with st.form("formulario_prediccion"):

        col1, col2 = st.columns(2)
        datos_estudiante = {}

        for indice, variable in enumerate(variables):

            opciones = sorted(
                df[variable]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            columna = col1 if indice % 2 == 0 else col2

            with columna:
                datos_estudiante[variable] = st.selectbox(
                    etiquetas.get(variable, variable),
                    opciones,
                    key=variable
                )

        enviar = st.form_submit_button(
            "Realizar predicción",
            use_container_width=True
        )

    if enviar:
        mostrar_aviso(
            "El formulario funciona correctamente. "
            "La predicción real se habilitará "
            "cuando integremos el modelo entrenado.",
            "Formulario recibido"
        )

# ==========================================
# 11. RESULTADOS
# ==========================================

elif pagina == "Resultados":

    st.title("🏆 Comparación de modelos")

    st.write(
        "En esta sección se presentarán los "
        "resultados de los tres modelos "
        "de Machine Learning."
    )

    mostrar_aviso(
        "Pendiente de integrar los resultados "
        "del entrenamiento y evaluación.",
        "Estado"
    )

    st.subheader("Evaluación de modelos")

    st.write(
        "Las métricas dependerán del tipo "
        "de problema predictivo definido."
    )

    st.write(
        "Para clasificación: Accuracy, "
        "Precision, Recall y F1-score."
    )

    st.write(
        "Para regresión: MAE, RMSE y R²."
    )

    st.subheader("Modelo seleccionado")

    st.write(
        "Aquí se mostrará el modelo con mejor "
        "desempeño y la justificación de "
        "su selección."
    )

    st.subheader("Conclusiones")

    st.write(
        "Las conclusiones finales se agregarán "
        "después de comparar los modelos."
    )

# ==========================================
# 12. PIE DE PÁGINA
# ==========================================

st.divider()

st.caption(
    "Universidad de San Buenaventura | "
    "Proyecto Final de Ciencia de Datos | 2026"
)
