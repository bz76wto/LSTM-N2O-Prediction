import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

# Load your dataset
data = pd.read_excel('/data/N2O_Data_2021.xlsx')

# Prepare the lagged features (today, yesterday, two days ago)
data['Temp_Yesterday'] = data['Avg Temperature'].shift(1)
data['Temp_Two_Days_Ago'] = data['Avg Temperature'].shift(2)
data['Rain_Yesterday'] = data['Avg Rainfall'].shift(1)
data['Rain_Two_Days_Ago'] = data['Avg Rainfall'].shift(2)
data['Emissions_Yesterday'] = data['N2O Emissions'].shift(1)
data['Emissions_Two_Days_Ago'] = data['N2O Emissions'].shift(2)

# Drop rows with missing values (from shifting)
data = data.dropna().reset_index(drop=True)

# Define features (X) and target (y)
X = data[['Avg Temperature', 'Avg Rainfall', 'Temp_Yesterday', 'Temp_Two_Days_Ago',
          'Rain_Yesterday', 'Rain_Two_Days_Ago', 'Emissions_Yesterday', 'Emissions_Two_Days_Ago']]
y = data['N2O Emissions']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the TDNN model
model = Sequential([
    Input(shape=(X_train.shape[1],)),    # Input layer matching feature count
    Dense(64, activation='relu'),        # Hidden layer 1
    Dense(32, activation='relu'),        # Hidden layer 2
    Dense(1)                             # Output layer for regression (N2O Emissions)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=1)

# Evaluate the model on test data
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print("Test Loss (MSE):", test_loss)
print("Test MAE:", test_mae)
