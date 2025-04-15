import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
import pandas as pd
crop= pd.read_csv('data/Crop_recommendation.csv')


# Label encoding
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}
crop['label_num'] = crop['label'].map(crop_dict)
crop.drop('label', axis=1, inplace=True)

# Features and labels
X = crop.drop('label_num', axis=1)
y = crop['label_num']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save model to file
with open('crop_recommendation_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

# Save scaler (needed for prediction)
with open('crop_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save label mapping for decoding predictions
with open('crop_label_mapping.pkl', 'wb') as file:
    pickle.dump(crop_dict, file)

print("✅ Model, scaler, and label mapping saved successfully.")
