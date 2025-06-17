from os import name
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(name)

# Load the model
model = joblib.load("random_forest-model.pkl")

# HTML Form Template
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            max-width: 500px;
            margin: auto;
            background: #f4f4f4;
        }
        h2 {
            text-align: center;
        }
        form {
            background: white;
            padding: 20px;
            border-radius: 10px;
        }
        label {
            display: block;
            margin-top: 15px;
        }
        input[type="number"] {
            width: 100%;
            padding: 8px;
            margin-top: 5px;
        }
        button {
            margin-top: 20px;
            padding: 10px;
            width: 100%;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
        }
        .result {
            margin-top: 20px;
            text-align: center;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h2>Iris Flower Prediction</h2>
    <form method="post" action="/predict">
        <label>Sepal Length (cm):</label>
        <input type="number" name="sepal_length" step="0.1" required>
        
        <label>Sepal Width (cm):</label>
        <input type="number" name="sepal_width" step="0.1" required>
        
        <label>Petal Length (cm):</label>
        <input type="number" name="petal_length" step="0.1" required>
        
        <label>Petal Width (cm):</label>
        <input type="number" name="petal_width" step="0.1" required>
        
        <button type="submit">Submit & Predict</button>
    </form>
    {% if prediction is not none %}
        <div class="result">Prediction: {{ prediction }}</div>
    {% endif %}
</body>
</html>
"""

# Route to serve form
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]
            pred = model.predict([features])[0]
            class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            prediction = class_names[int(pred)]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template_string(html_form, prediction=prediction)

# REST API Endpoint
@app.route("/predict", methods=["POST"])
def api_predict():
    if request.is_json:
        data = request.get_json()
        try:
            features = np.array(data["features"]).reshape(1, -1)
            prediction = model.predict(features)[0]
            return jsonify({"prediction": int(prediction)})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return home()  # Handle form submission

if name == "main":
    app.run(debug=True)