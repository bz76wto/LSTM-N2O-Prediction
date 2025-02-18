import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop, Nadam
import keras_tuner as kt
import matplotlib.pyplot as plt

# Function to prepare time-delay features (e.g., today, yesterday, two days ago)
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

# Combine datasets
combined_data = pd.concat([data_2021, data_2022]).reset_index(drop=True)

# Add sine/cosine transformations for DoY (Day of Year)
combined_data['DoY_sin'] = np.sin(2 * np.pi * combined_data['DoY'] / 365)
combined_data['DoY_cos'] = np.cos(2 * np.pi * combined_data['DoY'] / 365)

# Add rolling features for temperature
combined_data['Temp_Roll_Mean'] = combined_data['Avg Temperature'].rolling(window=7, min_periods=1).mean()
combined_data['Temp_Roll_Std'] = combined_data['Avg Temperature'].rolling(window=7, min_periods=1).std()

# Add temperature anomalies
combined_data['Temp_Anomaly'] = combined_data['Avg Temperature'] - combined_data['Avg Temperature'].mean()

# Add lagged temperature features
combined_data['Temp_Lag_1'] = combined_data['Avg Temperature'].shift(1)
combined_data['Temp_Lag_2'] = combined_data['Avg Temperature'].shift(2)

# Add rainfall categories and interactions
def categorize_rainfall(rain):
    if rain == 0: return 'none'
    elif rain < 5: return 'light'
    elif rain < 20: return 'moderate'
    else: return 'heavy'

combined_data['Rainfall_Category'] = combined_data['Avg Rainfall'].apply(categorize_rainfall)
combined_data = pd.get_dummies(combined_data, columns=['Rainfall_Category'])
combined_data['Rain_Temp_Interaction'] = combined_data['Avg Rainfall'] * combined_data['Avg Temperature']

# Add lagged rainfall features
combined_data['Rain_Lag_1'] = combined_data['Avg Rainfall'].shift(1)
combined_data['Rain_Lag_2'] = combined_data['Avg Rainfall'].shift(2)

# Add rolling mean for N2O emissions
combined_data['N2O_Roll_Mean'] = combined_data['N2O Emissions'].rolling(window=7, min_periods=1).mean()

# Define parameters
time_steps = 10 # Using a longer time window for LSTM
feature_cols = ['Avg Temperature', 'DoY_sin', 'DoY_cos', 'Temp_Roll_Mean', 'Temp_Roll_Std', 'Temp_Anomaly', 'Temp_Lag_1', 'Temp_Lag_2',
                'Rain_Temp_Interaction', 'Rain_Lag_1', 'Rain_Lag_2'] + \
               [col for col in combined_data.columns if 'Rainfall_Category' in col]
target_col = 'N2O Emissions'

# Drop rows with NaN values due to lagging
combined_data = combined_data.dropna()

# Prepare training and testing datasets
X, y = prepare_time_delay_data(combined_data, feature_cols, target_col, time_steps)

# Standardize input features
scaler = StandardScaler()
X_reshaped = X.reshape(X.shape[0], -1)  # Flatten for scaling
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape)  # Reshape back for LSTM

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the model builder function for LSTM
def build_lstm_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    
    # LSTM layers
    for i in range(hp.Int('num_lstm_layers', 1, 3)):
        model.add(LSTM(hp.Int(f'units_lstm_{i}', min_value=32, max_value=128, step=32),
                       return_sequences=(i != hp.Int('num_lstm_layers', 1, 3) - 1)))  # Last layer should not return sequences
        model.add(Dropout(hp.Float(f'dropout_lstm_{i}', 0.2, 0.5, step=0.1)))
    
    # Dense output layer
    model.add(Dense(1))

    # Optimizer choice
    optimizer_choice = hp.Choice('optimizer', ['RMSprop', 'Nadam'])
    optimizer = RMSprop(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])) if optimizer_choice == 'RMSprop' else Nadam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5]))

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model

# Initialize the tuner
tuner = kt.RandomSearch(
    build_lstm_model,
    objective='val_loss',
    max_trials=10,  # Reduce for faster tuning
    executions_per_trial=1,
    directory='C:/Users/ayonk095/Desktop/tdnn_tuning_results',
    project_name='lstm_tuning'
)

# Run the tuner
tuner.search(X_train_split, y_train_split, 
             validation_data=(X_val_split, y_val_split),
             epochs=50, batch_size=32)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and train the best model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=200, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
)

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Tuned LSTM Performance on Test Set:")
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
