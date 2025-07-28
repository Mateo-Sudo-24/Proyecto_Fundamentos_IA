from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# Aseguramos que el directorio de modelos exista
if not os.path.exists('../models'):
    os.makedirs('../models')

app = Flask(__name__, template_folder="templates")

# Cargar los modelos que ahora contienen el preprocesador
try:
    modelo_logistica = joblib.load('../models/pipeline_regresion_logistica.pkl')
    modelo_arbol = joblib.load('../models/pipeline_arbol_decision.pkl')
except Exception as e:
    print(f"Error al cargar los modelos: {e}")
    modelo_logistica = None
    modelo_arbol = None

def transformar_nivel_ingresos(valor):
    mapa = {'Alto': 2, 'Medio': 1, 'Bajo': 0}
    return mapa.get(valor, -1)  # -1 si no coincide

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    modelo_usado = None
    error_msg = None

    if request.method == 'POST':
        try:
            # Recopilar datos del formulario
            datos_formulario = {
                'sumValTot': [float(request.form['sumValTot'])],
                'Frecuencia': [int(request.form['frecuencia'])],
                'edad': [int(request.form['edad'])],
                'nivelIngresos': [transformar_nivel_ingresos(request.form['nivelIngresos'])],
                'estadoCivil': [request.form['estadoCivil']],
                'nivelEducacion': [request.form['nivelEducacion']],
                'sexo': [request.form['sexo']],
                'tipoVivienda': [request.form['tipoVivienda']]
            }
            # Crear DataFrame
            input_df = pd.DataFrame(datos_formulario)
            tipo_modelo = request.form['modelo']

            # Predicción
            if tipo_modelo == 'logistica' and modelo_logistica:
                prediccion_array = modelo_logistica.predict(input_df)
                prediccion = int(prediccion_array[0])
                modelo_usado = "Regresión Logística"
            elif tipo_modelo == 'arbol' and modelo_arbol:
                prediccion_array = modelo_arbol.predict(input_df)
                prediccion = int(prediccion_array[0])
                modelo_usado = "Árbol de Decisión"
            else:
                error_msg = "No se seleccionó un modelo válido o no se cargó correctamente."
        except Exception as e:
            error_msg = f"Error durante la predicción: {e}"
            print(error_msg)
            prediccion = None

    return render_template('index.html', prediccion=prediccion, modelo_usado=modelo_usado, error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)