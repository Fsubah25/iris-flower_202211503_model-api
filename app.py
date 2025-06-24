from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model_202211503.pkl', 'rb') as file:
    model = pickle.load(file)

# Feature names and target names
feature_names = ['sepal length (cm)', 'sepal width (cm)', 
                 'petal length (cm)', 'petal width (cm)']
target_names = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    return render_template('index.html', 
                         feature_names=feature_names,
                         target_names=target_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        if request.content_type == 'application/json':
            data = request.get_json()
            features = data['features']
        else:
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]
        
        # Convert to numpy array and ensure correct shape
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        predicted_class = int(prediction[0])
        predicted_species = target_names[predicted_class]
        
        # Return appropriate response
        if request.content_type == 'application/json':
            return jsonify({
                'predicted_class': predicted_class,
                'predicted_species': predicted_species,
                'features_used': dict(zip(feature_names, features[0]))
            })
        else:
            return render_template('index.html',
                                prediction=predicted_species,
                                feature_names=feature_names,
                                target_names=target_names,
                                feature_values=dict(zip(
                                    ['sepal_length', 'sepal_width', 
                                     'petal_length', 'petal_width'],
                                    features[0]
                                )))
    
    except Exception as e:
        if request.content_type == 'application/json':
            return jsonify({'error': str(e)}), 400
        else:
            return render_template('index.html',
                                  error=str(e),
                                  feature_names=feature_names,
                                  target_names=target_names)

if __name__ == '__main__':
    app.run(debug=True)