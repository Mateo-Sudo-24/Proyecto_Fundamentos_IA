from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Cargar ambos modelos
modelo_logistica = joblib.load('models/pipeline_regresion_logistica.pkl')
modelo_arbol = joblib.load('models/pipeline_arbol_decision.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    modelo_usado = None

    if request.method == 'POST':
        try:
            dato = float(request.form['dato'])
            tipo_modelo = request.form['modelo']
            
            if tipo_modelo == 'logistica':
                prediccion = modelo_logistica.predict([[dato]])
                modelo_usado = "Regresión Logística"
            elif tipo_modelo == 'arbol':
                prediccion = modelo_arbol.predict([[dato]])
                modelo_usado = "Árbol de Decisión"
        except Exception as e:
            prediccion = [f"Error: {e}"]

    return render_template('index.html', prediccion=prediccion, modelo_usado=modelo_usado)
