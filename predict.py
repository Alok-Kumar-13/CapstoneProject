import pandas as pd
import numpy as np
import pickle
import os
import mlflow

# Define ModelWrapper class (must match the one used during model saving)
# This class wraps the preprocessor (StandardScaler) and the model
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, scaler, model):
        # We store the scaler and the regressor model as attributes
        self.scaler = scaler
        self.model = model

    def predict(self, context, model_input):
        # This predict method is called by the MLflow pyfunc model.
        # It takes raw data, scales it using the stored scaler,
        # and then makes a prediction with the regressor model.
        scaled_input = self.scaler.transform(model_input)
        return self.model.predict(scaled_input)

# Define input data with correct types
input_data = pd.DataFrame([{
    'cylinders': np.int64(1),
    'displacement': np.float64(50.0),
    'horsepower': np.float64(950.0),
    'weight': np.int64(200),
    'acceleration': np.float64(6.0),
    'model year': np.int64(80),
    'origin_1': np.int32(1),
    'origin_2': np.int32(0),
    'origin_3': np.int32(0)
}])

# Path to model
model_path = "autompg_model.bin"

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Run prediction
try:
    predictions = model.predict(input_data)
    print("Prediction output:", predictions.tolist())
except Exception as e:
    print("Error during prediction:", e)
    raise