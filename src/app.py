# app.py

from flask import Flask, render_template, request
import joblib
import pandas as pd # <-- 1. Importar pandas

app = Flask(__name__)

# --- Carga de modelos con manejo de errores ---
try:
    modelo_logistica = joblib.load('models/pipeline_regresion_logistica.pkl')
    modelo_arbol = joblib.load('models/pipeline_arbol_decision.pkl')
except FileNotFoundError:
    print("FATAL: No se encontraron los archivos de modelo. La aplicación no podrá hacer predicciones.")
    modelo_logistica = None
    modelo_arbol = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion_final = None
    modelo_usado = None
    error_msg = None

    if request.method == 'POST':
        # Verificar que los modelos se cargaron
        if not modelo_logistica or not modelo_arbol:
            error_msg = "Error del servidor: Los modelos no están disponibles."
        else:
            try:
                # --- 2. Preparar los datos para la predicción ---
                # Reemplaza 'nombre_real_de_la_columna' con el nombre que usaste en el entrenamiento.
                # Ejemplo: si la columna era 'dias_credito', la línea sería: ['dias_credito']
                nombres_columnas = ['REEMPLAZAR_CON_NOMBRE_REAL'] # <-- ¡¡CRÍTICO!!

                dato_str = request.form['dato']
                if not dato_str: # Validar que el campo no esté vacío
                    raise ValueError("El campo de datos no puede estar vacío.")

                dato_float = float(dato_str)
                tipo_modelo = request.form['modelo']

                # Crear un DataFrame de Pandas con el nombre de columna correcto
                datos_para_predecir = pd.DataFrame([[dato_float]], columns=nombres_columnas)

                # --- 3. Realizar la predicción ---
                if tipo_modelo == 'logistica':
                    prediccion_array = modelo_logistica.predict(datos_para_predecir)
                    prediccion_final = prediccion_array[0] # Extraer el valor del array
                    modelo_usado = "Regresión Logística"
                elif tipo_modelo == 'arbol':
                    prediccion_array = modelo_arbol.predict(datos_para_predecir)
                    prediccion_final = prediccion_array[0] # Extraer el valor del array
                    modelo_usado = "Árbol de Decisión"

            except ValueError as ve:
                error_msg = f"Dato inválido: {ve}"
            except Exception as e:
                error_msg = f"Ocurrió un error inesperado: {e}"

    # 4. Pasar la variable de error a la plantilla
    return render_template('index.html', prediccion=prediccion_final, modelo_usado=modelo_usado, error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
