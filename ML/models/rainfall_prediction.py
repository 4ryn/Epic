import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, "..", "data", "rainfall_in_india_1901-2015.csv")
model_file_path = os.path.join(script_dir, "..", "scripts", "rainfall_prediction.pkl")

# Verify the file paths
if not os.path.exists(data_file_path):
    raise FileNotFoundError(f"Dataset not found: {data_file_path}")

# Load the dataset
df = pd.read_csv(data_file_path)

# Preprocess the data
# Columns 'SUBDIVISION', 'YEAR', and 'JUN' must exist in the dataset
required_columns = {'SUBDIVISION', 'YEAR', 'JUN'}
if not required_columns.issubset(df.columns):
    print(f"Error: Missing one or more required columns: {required_columns}")
    sys.exit(1)

# Features and target
X = df[['SUBDIVISION', 'YEAR']]
y = df['JUN']

# Handle missing values in the target variable y
y = y.fillna(y.mean())

# Preprocessing for categorical and numeric data
categorical_cols = ['SUBDIVISION']
numeric_cols = ['YEAR']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numeric_transformer, numeric_cols)])

# Append regressor to preprocessing pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model trained successfully. Mean Squared Error: {mse:.2f}")

# Save the model
joblib.dump(pipeline, model_file_path)
print("Model saved as 'rainfall_prediction.pkl'")

# Function to predict rainfall
def predict_rainfall(state, year):
    """
    Predict rainfall for a given state and year.
    Arguments:
        - state: str, state name
        - year: int, year for prediction
    Returns:
        - float: Predicted rainfall in mm
    """
    input_features = pd.DataFrame({'SUBDIVISION': [state], 'YEAR': [year]})
    try:
        prediction = pipeline.predict(input_features)
        return prediction[0]
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

# Main script
if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Command-line input
        state = sys.argv[1]
        try:
            year = int(sys.argv[2])
        except ValueError:
            print("Error: Year must be an integer.")
            sys.exit(1)

        try:
            predicted_rainfall = predict_rainfall(state, year)
            print(f"Predicted rainfall for {state} in year {year}: {predicted_rainfall:.2f} mm")
        except Exception as e:
            print(f"Error during prediction: {e}")
    else:
        # Interactive input
        print("\nEnter the following details for Rainfall Prediction:")
        try:
            state = input("State: ")
            year = int(input("Year: "))
            predicted_rainfall = predict_rainfall(state, year)
            print(f"\nPredicted Rainfall for {state} in year {year}: {predicted_rainfall:.2f} mm")
        except ValueError:
            print("Error: Year must be an integer.")
        except Exception as e:
            print(f"Error during rainfall prediction: {e}")
