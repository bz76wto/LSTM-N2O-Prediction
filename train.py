# train.py
from models.LSTM import build_lstm_model
from models.TDNN import build_tdnn_model
from utils.preprocess import load_data, add_features, normalize_data
from sklearn.model_selection import train_test_split

# Load Data
data_2021 = load_data("data/N2O_Data_2021.xlsx")
data_2022 = load_data("data/N2O_Data_2022.xlsx")

# Feature Engineering
data_2021 = add_features(data_2021)
data_2022 = add_features(data_2022)

# Define Features & Targets
feature_cols = ["Avg Temperature", "DoY_sin", "DoY_cos"]
target_col = "N2O Emissions"
X_train, y_train = data_2021[feature_cols], data_2021[target_col]
X_test, y_test = data_2022[feature_cols], data_2022[target_col]

# Normalize Data
X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)

# Split into Train/Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2)

# Train LSTM
lstm_model = build_lstm_model(input_shape=(5, len(feature_cols)))
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32)

# Train TDNN
tdnn_model = build_tdnn_model(input_shape=(X_train.shape[1],))
tdnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32)
