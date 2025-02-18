# visualize.py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import load_data, add_features, normalize_data

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

# Visualization
def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, label=f'{model_name} Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual N2O Emissions')
    plt.ylabel('Predicted N2O Emissions')
    plt.title(f'Actual vs Predicted N2O Emissions - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Results
plot_predictions(y_test, y_pred_lstm, "LSTM")
plot_predictions(y_test, y_pred_tdnn, "TDNN")
