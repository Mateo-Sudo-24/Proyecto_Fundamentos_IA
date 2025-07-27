from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import openai

app = Flask(__name__, template_folder="templates")

modelo_logistica = joblib.load('models/pipeline_regresion_logistica.pkl')
modelo_arbol = joblib.load('models/pipeline_arbol_decision.pkl')

# Cambia aquí según tu preferencia:
# openai.api_key = os.getenv("OPENAI_API_KEY")  # Si usas variable entorno

# O asigna directamente (sólo para pruebas rápidas):
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    modelo_usado = None

    if request.method == 'POST':
        try:
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
            input_df = pd.DataFrame(datos_formulario)

            tipo_modelo = request.form['modelo']

            if tipo_modelo == 'logistica':
                prediccion_array = modelo_logistica.predict(input_df)
                prediccion = prediccion_array[0]
                modelo_usado = "Regresión Logística"
            elif tipo_modelo == 'arbol':
                prediccion_array = modelo_arbol.predict(input_df)
                prediccion = prediccion_array[0]
                modelo_usado = "Árbol de Decisión"

        except Exception as e:
            print(f"Error durante la predicción: {e}")
            prediccion = -1

    return render_template('index.html', prediccion=prediccion, modelo_usado=modelo_usado)


@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        mensaje_usuario = data.get("message", "")

        prompt = f"""
Eres un asistente virtual experto en detección de actividades financieras sospechosas con base en criterios UAFE.
Usuario pregunta: {mensaje_usuario}
Responde de forma clara, concisa y útil.
Si no entiendes la pregunta, di "Lo siento, no entiendo tu pregunta. Por favor intenta con otra."
"""

        respuesta = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Cambia si quieres otro modelo
            messages=[
                {"role": "system", "content": "Eres un asistente experto en detección financiera."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            n=1,
            temperature=0.5,
        )

        texto_respuesta = respuesta.choices[0].message['content'].strip()

        return jsonify({"response": texto_respuesta})

    except Exception as e:
        print(f"Error en chatbot: {e}")
        return jsonify({"response": "⚠️ Hubo un error al procesar tu mensaje."})


if __name__ == '__main__':
    app.run(debug=True)
