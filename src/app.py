# app.py CORREGIDO

from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# Aseguramos que el directorio de modelos exista
if not os.path.exists('models'):
    os.makedirs('models')

app = Flask(__name__, template_folder="templates")

# Cargar los modelos que ahora contienen el preprocesador
# Asegúrate de que estos archivos .pkl existan en la carpeta 'models'
try:
    modelo_logistica = joblib.load('models/pipeline_regresion_logistica.pkl')
    modelo_arbol = joblib.load('models/pipeline_arbol_decision.pkl')
except FileNotFoundError as e:
    print(f"Error al cargar un modelo: {e}")
    # Considera manejar este error de manera más elegante en un entorno de producción
    modelo_logistica = None
    modelo_arbol = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    modelo_usado = None

    if request.method == 'POST':
        try:
            # 1. Recopilar todos los datos del formulario con los nombres correctos
            datos_formulario = {
                'sumValTot': [float(request.form['sumValTot'])],
                'Frecuencia': [int(request.form['frecuencia'])],
                'edad': [int(request.form['edad'])],
                'nivelIngresos': [request.form['nivelIngresos']],
                'estadoCivil': [request.form['estadoCivil']],
                'nivelEducacion': [request.form['nivelEducacion']],
                'sexo': [request.form['sexo']],
                'tipoVivienda': [request.form['tipoVivienda']]
            }
            
            # 2. Crear un DataFrame de Pandas
            input_df = pd.DataFrame(datos_formulario)
            
            tipo_modelo = request.form['modelo']
            
            # 3. Hacer la predicción con el DataFrame
            # Verificamos que los modelos se hayan cargado correctamente
            if tipo_modelo == 'logistica' and modelo_logistica:
                prediccion_array = modelo_logistica.predict(input_df)
                prediccion = int(prediccion_array[0])
                modelo_usado = "Regresión Logística"
            elif tipo_modelo == 'arbol' and modelo_arbol:
                prediccion_array = modelo_arbol.predict(input_df)
                prediccion = int(prediccion_array[0])
                modelo_usado = "Árbol de Decisión"
            
            # Si no se seleccionó un modelo válido o no se cargaron, la predicción será None
                
        except Exception as e:
            # Puedes manejar el error de forma más específica si quieres
            print(f"Error durante la predicción: {e}")
            prediccion = -1 # Un valor para indicar error

    return render_template('index.html', prediccion=prediccion, modelo_usado=modelo_usado)

if __name__ == '__main__':
    # Aquí puedes cambiar host a '0.0.0.0' para acceder desde cualquier IP
    app.run(debug=True, host='127.0.0.1', port=5000)