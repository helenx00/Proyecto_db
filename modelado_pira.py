# ==========================================================
# Parte de Pira: preparación, 3 modelos e hiperparámetros
# Problema: REGRESIÓN (predecir punt_global, un número 0-500)
# Datos: data/saber11_modelado.csv  (rama sugerida: pira-modelado)
# ==========================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Cargar datos
df = pd.read_csv("data/saber11_modelado.csv")
X = df.drop(columns="punt_global")
y = df["punt_global"]

# 2. Separar train / test ANTES de imputar (evita fuga de información,
#    tal como lo dejó planteado la líder en el notebook)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Tipos de variables
# Estrato y educación de los padres tienen orden -> codificación ordinal
orden_estrato = ["Sin Estrato", "Estrato 1", "Estrato 2", "Estrato 3",
                 "Estrato 4", "Estrato 5", "Estrato 6"]
orden_educ = ["Ninguno", "Primaria incompleta", "Primaria completa",
              "Secundaria (Bachillerato) incompleta",
              "Secundaria (Bachillerato) completa",
              "Técnica o tecnológica incompleta",
              "Técnica o tecnológica completa",
              "Educación profesional incompleta",
              "Educación profesional completa", "Postgrado"]
# "No sabe" y "No Aplica" no tienen posición en la escala -> se tratan
# como valores desconocidos (NaN) y se imputan.
ordinales = ["fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre"]
categoricas = ["fami_tieneinternet", "fami_tienecomputador",
               "cole_naturaleza", "cole_area_ubicacion", "estu_genero"]

def limpiar_desconocidos(d):
    return d.replace({"No sabe": np.nan, "No Aplica": np.nan})

X_train = limpiar_desconocidos(X_train)
X_test = limpiar_desconocidos(X_test)

prep_ord = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ord", OrdinalEncoder(
        categories=[orden_estrato, orden_educ, orden_educ],
        handle_unknown="use_encoded_value", unknown_value=-1)),
])
prep_cat = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
])
preparacion = ColumnTransformer([
    ("ord", prep_ord, ordinales),
    ("cat", prep_cat, categoricas),
])

# 4. Tres modelos + hiperparámetros a probar
modelos = {
    "Ridge (regresión lineal regularizada)": (
        Ridge(), {"modelo__alpha": [0.1, 1, 10, 100]}),
    "Random Forest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"modelo__n_estimators": [100, 300],
         "modelo__max_depth": [5, 10, None],
         "modelo__min_samples_leaf": [5, 20]}),
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=42),
        {"modelo__n_estimators": [100, 200],
         "modelo__learning_rate": [0.05, 0.1],
         "modelo__max_depth": [2, 3]}),
}

# 5. Entrenar con validación cruzada (solo con train) y evaluar en test
resultados = []
mejores = {}
for nombre, (modelo, grid) in modelos.items():
    pipe = Pipeline([("prep", preparacion), ("modelo", modelo)])
    gs = GridSearchCV(pipe, grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X_train, y_train)
    pred = gs.predict(X_test)
    resultados.append({
        "Modelo": nombre,
        "Mejores hiperparámetros": gs.best_params_,
        "RMSE (CV train)": -gs.best_score_,
        "MAE test": mean_absolute_error(y_test, pred),
        "RMSE test": np.sqrt(mean_squared_error(y_test, pred)),
        "R2 test": r2_score(y_test, pred),
    })
    mejores[nombre] = gs.best_estimator_

# Línea base: predecir siempre el promedio
base = np.full(len(y_test), y_train.mean())
print("Línea base (promedio): RMSE test =", round(np.sqrt(mean_squared_error(y_test, base)), 2))

tabla = pd.DataFrame(resultados)
pd.set_option("display.width", 200, "display.max_colwidth", 70)
print(tabla.round(3).to_string(index=False))
