from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import sys

app = Flask(__name__)

# --- File Paths Setup --- #
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# data_file_path = os.path.join(script_dir, "..", "data", "pump_data.csv")
model_file_path = os.path.join(script_dir,"pump_activation_model.pkl")

# --- Prediction Function --- #
def predict(data):
    """
    Predict pump activation (0=Off, 1=On) using the trained model.
    :param data: A dictionary or DataFrame containing feature values.
    :return: Predicted pump status (0 or 1).
    """
    # Load the model
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    
    model = joblib.load(model_file_path)

    # Convert input data to DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame(data, index=[0])
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a dictionary or DataFrame.")

    # Ensure correct feature columns
    expected_columns = ["Soil Moisture", "Temperature", "Air Humidity"]
    if not set(expected_columns).issubset(data.columns):
        raise ValueError(f"Input data must contain these columns: {expected_columns}")

    # Make prediction
    prediction = model.predict(data)
    return prediction[0]

@app.route('/')
def home():
    return render_template('indexw.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        # Get form data
        soil_moisture = float(request.form['soil_moisture'])
        temperature = float(request.form['temperature'])
        air_humidity = float(request.form['air_humidity'])
        
        # Prepare input data
        input_data = {
            "Soil Moisture": soil_moisture,
            "Temperature": temperature,
            "Air Humidity": air_humidity
        }
        
        # Make prediction
        result = predict(input_data)
        
        # Return result
        return render_template('indexw.html', 
                              prediction=int(result), 
                              soil_moisture=soil_moisture,
                              temperature=temperature,
                              air_humidity=air_humidity)
    
    except Exception as e:
        return render_template('indexw.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Prepare input data
        input_data = {
            "Soil Moisture": float(data['soil_moisture']),
            "Temperature": float(data['temperature']),
            "Air Humidity": float(data['air_humidity'])
        }
        
        # Make prediction
        result = predict(input_data)
        
        # Return result as JSON
        return jsonify({
            'success': True,
            'prediction': int(result),
            'message': f"Pump Activation Prediction: {result} (0=Off, 1=On)"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)