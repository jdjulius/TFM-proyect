from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load("model.pkl")

    data = request.get_json(force=True)
    features = data['features']
    
    df = pd.DataFrame([features])

    print(df)
    
    prediction = model.predict(df)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)