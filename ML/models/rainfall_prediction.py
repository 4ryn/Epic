from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.figure import Figure

app = Flask(__name__)

# Function to predict rainfall (same as in your original script)
def predict_rainfall(model, latitude, longitude, date_str):
    """
    Predict rainfall for a given location and date.
    
    Arguments:
        - model: Trained model
        - latitude: float, latitude coordinate
        - longitude: float, longitude coordinate
        - date_str: str, date in DD-MM-YYYY format
    
    Returns:
        - float: Predicted rainfall in mm
    """
    try:
        # Parse the date
        date = datetime.strptime(date_str, '%d-%m-%Y')
        month = date.month
        day = date.day
        
        # Create input features
        input_features = pd.DataFrame({
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Month': [month],
            'Day': [day]
        })
        
        # Make prediction
        prediction = model.predict(input_features)
        return max(0, prediction[0])  # Rainfall can't be negative
    
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

# Function to create a rainfall gauge visualization
def create_rainfall_gauge(rainfall_value):
    # Create a figure
    fig = Figure(figsize=(6, 1))
    ax = fig.subplots()
    
    # Define rainfall intensity categories
    max_display = max(30, rainfall_value * 1.2)  # Scale appropriately
    
    # Create color gradient for the gauge
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(0, max_display)
    
    # Plot horizontal bar
    ax.barh(0, max_display, color='lightgray', height=0.5)
    ax.barh(0, rainfall_value, color=cmap(norm(rainfall_value)), height=0.5)
    
    # Add labels and customize
    ax.set_xlim(0, max_display)
    ax.set_yticks([])
    ax.set_xticks([0, max_display/4, max_display/2, 3*max_display/4, max_display])
    ax.set_xlabel('Rainfall (mm)')
    ax.tick_params(axis='x', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Convert BytesIO to base64 string for HTML embedding
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

# Function to create a forecast line plot
def create_forecast_plot(dates, predictions):
    fig = Figure(figsize=(10, 5))
    ax = fig.subplots()
    
    # Plot the data
    ax.plot(dates, predictions, marker='o', linestyle='-', color='royalblue')
    
    # Format the plot
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Rainfall (mm)')
    ax.set_title('Rainfall Forecast')
    fig.autofmt_xdate()  # Rotate date labels
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Convert BytesIO to base64 string for HTML embedding
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

# Load the model
def load_model(model_path):
    """Load the trained model from file"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        print(f"Warning: Model file not found at {model_path}")
        return None

# Set the default model path
script_dir = os.path.dirname(os.path.abspath(__file__))
default_model_path = os.path.join(script_dir, "rainfall_prediction_model.pkl")
model = load_model(default_model_path)

@app.route('/')
def index():
    return render_template('index.html', model_loaded=(model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure the model file exists.'}), 400
    
    try:
        # Get form data
        latitude = float(request.form.get('latitude'))
        longitude = float(request.form.get('longitude'))
        date_str = request.form.get('date')  # Should be in DD-MM-YYYY format
        
        # Make prediction
        predicted_rainfall = predict_rainfall(model, latitude, longitude, date_str)
        
        # Determine rainfall intensity
        if predicted_rainfall < 2.5:
            intensity = "Light Rain (< 2.5 mm)"
            advice = "Light rain, minimal impact on agriculture."
        elif predicted_rainfall < 7.5:
            intensity = "Moderate Rain (2.5-7.5 mm)"
            advice = "Moderate rainfall, good for most crops."
        elif predicted_rainfall < 15:
            intensity = "Heavy Rain (7.5-15 mm)"
            advice = "Heavy rainfall, monitor drainage in fields."
        else:
            intensity = "Very Heavy Rain (> 15 mm)"
            advice = "Very heavy rainfall, potential flooding risk."
        
        # Create visualization
        gauge_img = create_rainfall_gauge(predicted_rainfall)
        
        return render_template(
            'result.html',
            rainfall=round(predicted_rainfall, 2),
            latitude=latitude,
            longitude=longitude,
            date=date_str,
            intensity=intensity,
            advice=advice,
            gauge_img=gauge_img
        )
        
    except Exception as e:
        return render_template('index.html', error=str(e), model_loaded=(model is not None))

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure the model file exists.'}), 400
    
    try:
        # Get form data
        latitude = float(request.form.get('latitude'))
        longitude = float(request.form.get('longitude'))
        start_date_str = request.form.get('start_date')  # Should be in DD-MM-YYYY format
        end_date_str = request.form.get('end_date')  # Should be in DD-MM-YYYY format
        
        # Parse dates
        start_date = datetime.strptime(start_date_str, '%d-%m-%Y')
        end_date = datetime.strptime(end_date_str, '%d-%m-%Y')
        
        if start_date > end_date:
            return render_template('index.html', error="End date must be after start date", model_loaded=(model is not None))
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date)
        date_strs = [date.strftime("%d-%m-%Y") for date in date_range]
        
        # Make predictions for each date
        predictions = []
        for date_str in date_strs:
            try:
                rain = predict_rainfall(model, latitude, longitude, date_str)
                predictions.append(rain)
            except Exception as e:
                return render_template('index.html', error=f"Error predicting for {date_str}: {str(e)}", model_loaded=(model is not None))
        
        # Create forecast table data
        forecast_data = []
        for i, date in enumerate(date_range):
            forecast_data.append({
                'date': date.strftime("%d-%m-%Y"),
                'rainfall': round(predictions[i], 2)
            })
        
        # Create visualization
        forecast_img = create_forecast_plot(date_range, predictions)
        
        return render_template(
            'forecast.html',
            forecast_data=forecast_data,
            latitude=latitude,
            longitude=longitude,
            start_date=start_date_str,
            end_date=end_date_str,
            forecast_img=forecast_img
        )
        
    except Exception as e:
        return render_template('index.html', error=str(e), model_loaded=(model is not None))

@app.route('/set_model', methods=['POST'])
def set_model():
    try:
        global model
        model_path = request.form.get('model_path')
        model = load_model(model_path)
        
        if model is None:
            return render_template('index.html', error=f"Model not found at {model_path}", model_loaded=False)
        
        return render_template('index.html', success=f"Model loaded successfully from {model_path}", model_loaded=True)
    
    except Exception as e:
        return render_template('index.html', error=str(e), model_loaded=(model is not None))

if __name__ == '__main__':
    app.run(debug=True)