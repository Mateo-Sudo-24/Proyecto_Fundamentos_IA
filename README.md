Generated markdown
# Repositorio de Despliegue - Modelo Predictivo de Actividad Financiera Sospechosa

Este repositorio contiene la aplicación y los componentes necesarios para **desplegar y utilizar** el modelo predictivo de actividad financiera sospechosa. Su función es cargar un artefacto de modelo pre-entrenado (`.pkl`) y utilizarlo para realizar predicciones sobre nuevos datos en un entorno de producción.

**Este repositorio no contiene el código de entrenamiento.**

## 1. Descripción del Servicio

Este servicio expone el modelo de Machine Learning para que pueda ser consumido por otras aplicaciones. Recibe datos de clientes en un formato específico, los procesa a través del pipeline del modelo y devuelve una predicción (sospechosa o no sospechosa) junto con una puntuación de probabilidad.

## 2. Estructura del Repositorio de Despliegue

- **`/` (raíz)**:
  - **`predict.py` (o `app.py`)**: Script principal para cargar el modelo y realizar predicciones. Puede ser un script de línea de comandos o una API web (usando Flask, FastAPI, etc.).
  - **`requirements.txt`**: Un listado de las dependencias de Python necesarias para ejecutar el script de predicción.
  - **`README.md`**: Este archivo.
- **`/models/`**:
  - **`pipeline_arbol_decision.pkl`**: El artefacto del modelo entrenado. (Se puede elegir cualquiera de los modelos generados en el repositorio de entrenamiento).

## 3. Guía de Inicio y Uso

### Prerrequisitos
- Python 3.8 o superior.
- Entorno virtual recomendado.

### Pasos para la Instalación
1.  **Clonar el repositorio:**
    ```bash
    git clone [URL-DEL-REPOSITORIO-DE-DESPLIEGUE]
    cd [NOMBRE-DEL-REPOSITORIO]
    ```
2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

### Cómo Realizar una Predicción

El siguiente es un ejemplo de cómo podría ser el script `predict.py` para cargar el modelo y usarlo.

**Archivo `app.py`:**
```python
import pandas as pd
import joblib

# Cargar el pipeline del modelo desde el directorio /models
MODEL_PATH = 'models/pipeline_arbol_decision.pkl'
try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Archivo de modelo no encontrado en {MODEL_PATH}")

def predecir_actividad(datos_cliente: pd.DataFrame) -> dict:
    """
    Realiza una predicción sobre los datos de un cliente.
    
    Args:
        datos_cliente (pd.DataFrame): DataFrame con los datos del cliente.
    
    Returns:
        dict: Un diccionario con la predicción y la probabilidad.
    """
    # El pipeline se encarga del preprocesamiento internamente
    prediccion = pipeline.predict(datos_cliente)[0]
    probabilidades = pipeline.predict_proba(datos_cliente)[0]
    
    resultado = {
        'prediccion': int(prediccion),
        'etiqueta': "Sospechosa" if prediccion == 1 else "No Sospechosa",
        'probabilidad_no_sospechosa': probabilidades[0],
        'probabilidad_sospechosa': probabilidades[1]
    }
    
    return resultado

if __name__ == '__main__':
    # --- Ejemplo de Uso ---
    # Crear un DataFrame con los datos de un nuevo cliente
    # Las columnas deben coincidir con las usadas en el entrenamiento
    nuevo_cliente = pd.DataFrame({
        'sumValTot': [15000.50],
        'Frecuencia': [20],
        'edad': [35.0],
        'estadoCivil': ['S'],
        'nivelEducacion': ['Universitaria'],
        'sexo': ['F'],
        'tipoVivienda': ['Propia'],
        'nivelIngresos': [3200.0]
    })

    # Obtener la predicción
    resultado_prediccion = predecir_actividad(nuevo_cliente)

    # Mostrar el resultado
    print("--- Resultado de la Predicción ---")
    print(f"  - Etiqueta Predicha: {resultado_prediccion['etiqueta']}")
    print(f"  - Valor Predicho: {resultado_prediccion['prediccion']}")
    print(f"  - Probabilidad de ser Sospechosa: {resultado_prediccion['probabilidad_sospechosa']:.2%}")
```


Formato de Entrada y Salida
Entrada: Un DataFrame de Pandas con las columnas requeridas (sumValTot, Frecuencia, edad, estadoCivil, nivelEducacion, sexo, tipoVivienda, nivelIngresos).
Salida: Un diccionario que contiene:
prediccion: 0 para "No Sospechosa", 1 para "Sospechosa".
etiqueta: La etiqueta de texto correspondiente.
probabilidad_sospechosa: Un valor flotante entre 0 y 1 que indica la confianza del modelo en la predicción de "Sospechosa".

