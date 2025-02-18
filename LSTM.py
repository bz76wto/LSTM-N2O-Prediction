import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Function to prepare time-delay features with additional input features
def prepare_time_delay_data(data, feature_cols, target_col, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        feature_window = data[feature_cols].iloc[i:i + time_steps].values
        X.append(feature_window)
        y.append(data[target_col].iloc[i + time_steps])
    return np.array(X), np.array(y)

# Load datasets
data_2021 = pd.read_excel('C:/Users/ayonk095/Desktop/Test/N2O_Data_2022.xlsx')
data_2022 = pd.read_excel('C:/Users/ayonk095/Desktop/Test/N2O_Data_2021.xlsx')

# Add sine/cosine transformations for DoY (Day of Year)
data_2021['DoY_sin'] = np.sin(2 * np.pi * data_2021['Avg Temperature'] / 365)
data_2021['DoY_cos'] = np.cos(2 * np.pi * data_2021['Avg Temperature'] / 365)
data_2022['DoY_sin'] = np.sin(2 * np.pi * data_2022['Avg Temperature'] / 365)
data_2022['DoY_cos'] = np.cos(2 * np.pi * data_2022['Avg Temperature'] / 365)
data_2021['Temp_Roll_Mean'] = data_2021['Avg Temperature'].rolling(window=3).mean()
data_2022['Temp_Roll_Mean'] = data_2022['Avg Temperature'].rolling(window=3).mean()

# Sort datasets by DoY for sequential order
data_2021_sorted = data_2021.sort_values(by='DoY').reset_index(drop=True)
data_2022_sorted = data_2022.sort_values(by='DoY').reset_index(drop=True)

# Define parameters
time_steps = 5  # Extended to 5 days
feature_cols = ['Avg Temperature','DoY_sin', 'DoY_cos']  # Include additional input features
target_col = 'N2O Emissions'

# Prepare training (2021) and test (2022) datasets
X_train, y_train = prepare_time_delay_data(data_2021_sorted, feature_cols, target_col, time_steps)
X_test, y_test = prepare_time_delay_data(data_2022_sorted, feature_cols, target_col, time_steps)

# Standardize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Split training data into train and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(128, activation='tanh', return_sequences=True, input_shape=(time_steps, len(feature_cols))),
    Dropout(0.3),
    LSTM(64, activation='tanh'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for N2O Emissions
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=200, batch_size=32, callbacks=[early_stopping], verbose=1
)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print(f"Tuned LSTM Model Performance on Test Set:")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-Squared (R2): {r2:.3f}")

# Visualize Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual N2O Emissions')
plt.ylabel('Predicted N2O Emissions')
plt.title('Actual vs Predicted N2O Emissions')
plt.legend()
plt.grid(True)
plt.show()

# Visualize Residuals
residuals = y_test - y_pred.flatten()
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
