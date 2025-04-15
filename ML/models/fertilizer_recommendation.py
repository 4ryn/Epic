import pandas as pd
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# File paths
data_file_path = os.path.join(script_dir, "..", "data", "fertilizer_recommendation.csv")
model_file_path = os.path.join(script_dir, "..", "scripts", "fertilizer_recommendation.pkl")
encoder_file_path = os.path.join(script_dir, "..", "scripts", "label_encoders.pkl")

# Verify the file paths
if not os.path.exists(data_file_path):
    raise FileNotFoundError(f"Dataset not found: {data_file_path}")

# Load the dataset
data = pd.read_csv(data_file_path)
Soil_Type = data['Soil Type'].unique()
Crop_Type = data['Crop Type'].unique()
# print("Soil Types:", Soil_Type)
# print("Crop Types:", Crop_Type)
# Label encoding for categorical features
le_soil = LabelEncoder()
le_crop = LabelEncoder()

# Fit LabelEncoders on all possible values
data['Soil Type'] = le_soil.fit_transform(data['Soil Type'])
data['Crop Type'] = le_crop.fit_transform(data['Crop Type'])

# Save encoders for later use
encoders = {'soil': le_soil, 'crop': le_crop}
joblib.dump(encoders, encoder_file_path)

# Splitting the data into input and output variables
X = data[['Temparature', 'Humidity', 'Soil Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Fertilizer Name']

# Training the Decision Tree Classifier model
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X, y)

# Save the trained model
joblib.dump(dtc, model_file_path)
print("Model and encoders have been saved successfully.")

# Get the input parameters from command line arguments
if len(sys.argv) != 9:
    print("Usage: python fertilizer_recommendation.py <Temperature> <Humidity> <Soil_Moisture> <Soil_Type> <Crop_Type> <Nitrogen> <Potassium> <Phosphorous>")
    sys.exit(1)

Temperature = float(sys.argv[1])
Humidity = float(sys.argv[2])
Soil_Moisture = float(sys.argv[3])
Soil_Type = sys.argv[4]
Crop_Type = sys.argv[5]
Nitrogen = float(sys.argv[6])
Potassium = float(sys.argv[7])
Phosphorous = float(sys.argv[8])

# Load the encoders and model
encoders = joblib.load(encoder_file_path)
dtc = joblib.load(model_file_path)

# Handle unseen labels for Soil_Type and Crop_Type
try:
    soil_enc = encoders['soil'].transform([Soil_Type])[0]
except ValueError:
    print(f"Error: Soil Type '{Soil_Type}' is not recognized.")
    sys.exit(1)

try:
    crop_enc = encoders['crop'].transform([Crop_Type])[0]
except ValueError:
    print(f"Error: Crop Type '{Crop_Type}' is not recognized.")
    sys.exit(1)

# Prepare the user input for prediction
user_input = [[Temperature, Humidity, Soil_Moisture, soil_enc, crop_enc, Nitrogen, Potassium, Phosphorous]]

# Predict the fertilizer
fertilizer_name = dtc.predict(user_input)

# Return the prediction as a string
print(f"Recommended Fertilizer: {fertilizer_name[0]}")


