# NUEVO CÓDIGO DE PREPROCESAMIENTO Y ENTRENAMIENTO

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

# 1. Definir las características ANTES del encoding
X = df.drop(columns=['Uafe', 'codigoCliente'])
y = df['Uafe']

# 2. Identificar tipos de columnas
numeric_features = ['sumValTot', 'Frecuencia', 'edad', 'nivelIngresos']
categorical_features = ['estadoCivil', 'nivelEducacion', 'sexo', 'tipoVivienda']

# 3. Crear el preprocesador con ColumnTransformer
#    - Escala las numéricas
#    - Aplica One-Hot Encoding a las categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Dividir los datos ANTES de cualquier transformación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Modelo de Regresión Logística ---
pipeline_r = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(max_iter=1000))])

print("Entrenando Regresión Logística...")
pipeline_r.fit(X_train, y_train)
joblib.dump(pipeline_r, 'models/pipeline_regresion_logistica.pkl')
print("¡Pipeline de Regresión Logística guardado!")


# --- Modelo de Árbol de Decisión ---
pipeline_dt = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))])

print("\nEntrenando Árbol de Decisión...")
pipeline_dt.fit(X_train, y_train)
joblib.dump(pipeline_dt, 'models/pipeline_arbol_decision.pkl')
print("¡Pipeline de Árbol de Decisión guardado!")
