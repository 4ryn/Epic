import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# File paths
data_file_path = os.path.join(script_dir, "..", "data", "crop_production_MADHYA PRADESH.csv")
model_file_path = os.path.join(script_dir, "..", "scripts", "yield_prediction_model.pkl")
encoder_file_path = os.path.join(script_dir, "..", "scripts", "yield_prediction_ohe.pkl")

# Verify the file paths
if not os.path.exists(data_file_path):
    raise FileNotFoundError(f"Dataset not found: {data_file_path}")
if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"Model file not found: {model_file_path}")
if not os.path.exists(encoder_file_path):
    raise FileNotFoundError(f"Encoder file not found: {encoder_file_path}")

# Load the dataset
df = pd.read_csv(data_file_path)

# Process dataset
df = df.drop(['Crop_Year'], axis=1)
X = df.drop(['Production'], axis=1)
y = df['Production']

# Load models and encoder
model = joblib.load(model_file_path)
ohe = joblib.load(encoder_file_path)

# Prediction function
def predict_yield(state, district, season, crop, area):
    user_input = pd.DataFrame({
        'State_Name': [state],
        'District_Name': [district],
        'Season': [season],
        'Crop': [crop],
        'Area': [area]
    })

    categorical_cols = ['State_Name', 'District_Name', 'Season', 'Crop']
    user_input_categorical = ohe.transform(user_input[categorical_cols])
    user_input_numeric = user_input.drop(categorical_cols, axis=1)

    user_input_final = np.hstack((user_input_categorical.toarray(), user_input_numeric))
    prediction = model.predict(user_input_final)
    return prediction[0]

# Main script
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 6:
        state, district, season, crop, area = sys.argv[1:6]
        area = float(area)
        try:
            yield_prediction = predict_yield(state, district, season, crop, area)
            print(f"Predicted Yield: {yield_prediction} tons")
        except Exception as e:
            print(f"Error during yield prediction: {e}")
    else:
        print("Usage: python yield_prediction.py <state> <district> <season> <crop> <area>")


import joblib

# Save the trained model
joblib.dump(model, os.path.join(script_dir, "..", "scripts", "yield_prediction_model.pkl"))

# Save the OneHotEncoder
joblib.dump(ohe, os.path.join(script_dir, "..", "scripts", "yield_prediction_ohe.pkl"))

print("Model and encoder have been saved successfully.")
