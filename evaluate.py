# evaluate.py
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import load_data, add_features, normalize_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
data_2022 = load_data("data/N2O_Data_2022.xlsx")
data_2022 = add_features(data_2022)

# Define Features & Targets
feature_cols = ["Avg Temperature", "DoY_sin", "DoY_cos"]
target_col = "N2O Emissions"
X_test, y_test = data_2022[feature_cols], data_2022[target_col]

# Normalize Data
X_test_scaled, _ = normalize_data(X_test, X_test)

# Load Models
lstm_model = load_model("models/lstm_model.h5")
tdnn_model = load_model("models/tdnn_model.h5")

# Predictions
y_pred_lstm = lstm_model.predict(X_test_scaled)
y_pred_tdnn = tdnn_model.predict(X_test_scaled)

# Evaluate Performance
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.3f}, MSE: {mse:.3f}, R²: {r2:.3f}\n")

# Print Results
evaluate_model(y_test, y_pred_lstm, "LSTM")
evaluate_model(y_test, y_pred_tdnn, "TDNN")
