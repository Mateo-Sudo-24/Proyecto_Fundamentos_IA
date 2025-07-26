# app.py CORREGIDO

from flask import Flask, render_template, request
import joblib
import pandas as pd

# ...existing code...
app = Flask(__name__, template_folder="templates")


# ...existing code...ar los modelos que ahora contienen el preprocesador
modelo_logistica = joblib.load('models/pipeline_regresion_logistica.pkl')
modelo_arbol = joblib.load('models/pipeline_arbol_decision.pkl')
# ...existing code...

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    modelo_usado = None

    if request.method == 'POST':
        try:
            # 1. Recopilar todos los datos del formulario
            datos_formulario = {
                'sumValTot': [float(request.form['sumValTot'])],
                'Frecuencia': [int(request.form['Frecuencia'])],
                'edad': [int(request.form['edad'])],
                'nivelIngresos': [float(request.form['nivelIngresos'])],
                'estadoCivil': [request.form['estadoCivil']],
                'nivelEducacion': [request.form['nivelEducacion']],
                'sexo': [request.form['sexo']],
                'tipoVivienda': [request.form['tipoVivienda']]
            }
            
            # 2. Crear un DataFrame de Pandas
            input_df = pd.DataFrame(datos_formulario)
            
            tipo_modelo = request.form['modelo']
            
            # 3. Hacer la predicción con el DataFrame
            # El pipeline se encargará del escalado y el one-hot encoding automáticamente
            if tipo_modelo == 'logistica':
                prediccion_array = modelo_logistica.predict(input_df)
                prediccion = prediccion_array[0]
                modelo_usado = "Regresión Logística"
            elif tipo_modelo == 'arbol':
                prediccion_array = modelo_arbol.predict(input_df)
                prediccion = prediccion_array[0]
                modelo_usado = "Árbol de Decisión"
                
        except Exception as e:
            # Puedes manejar el error de forma más específica si quieres
            print(f"Error durante la predicción: {e}")
            prediccion = -1 # Un valor para indicar error

    return render_template('index.html', prediccion=prediccion, modelo_usado=modelo_usado)

if __name__ == '__main__':
    app.run(debug=True)
