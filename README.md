# 🌍 LSTM for Nitrous Oxide (N₂O) Emissions Prediction

## 🚀 Project Overview
This repository contains an **LSTM-based deep learning model** designed to predict **Nitrous Oxide (N₂O) emissions** using historical **temperature and seasonal data**. The model leverages time-series forecasting techniques to analyze and predict emissions based on environmental factors.

## 📊 Dataset & Features
### **1. Input Data**
The dataset consists of:
- **Temperature Data** (`Avg Temperature`)
- **Day of the Year (DoY) Transformations** (`sin(DoY)`, `cos(DoY)`) to capture seasonal cycles
- **Rolling Mean Temperature** (3-day moving average to smooth fluctuations)
- **Target Variable**: `N2O Emissions`

### **2. Data Preprocessing**
- **Cyclical Encoding:** Converts day-of-year (`DoY`) to **sine & cosine values** for seasonality.
- **Sliding Window:** Uses a **5-day sequence** to predict the next day's N₂O emissions.
- **Standardization:** Features are normalized using `StandardScaler`.

## 🧠 LSTM Model Architecture
Built using `TensorFlow/Keras`, the model consists of:
- 🔹 **LSTM Layer (128 units, tanh activation, return sequences=True)**
- 🔹 **Dropout Layer (30%)**
- 🔹 **LSTM Layer (64 units, tanh activation)**
- 🔹 **Dropout Layer (30%)**
- 🔹 **Dense Layer (32 units, ReLU activation)**
- 🔹 **Dense Output Layer (1 unit, Linear activation)** (Predicts N₂O emissions)

### **Training Configuration**
- **Optimizer:** `Adam`
- **Loss Function:** `Mean Squared Error (MSE)`
- **Metrics:** `Mean Absolute Error (MAE)`
- **Early Stopping:** Stops training if validation loss does not improve for 20 epochs.

## 📈 Model Performance & Evaluation
- **MAE (Mean Absolute Error)**: Measures prediction error.
- **MSE (Mean Squared Error)**: Penalizes large errors.
- **R² Score**: Indicates how well the model explains variance.

### **Visualizations**
1. **Actual vs. Predicted Scatter Plot** - Compares true vs. predicted emissions.
2. **Residuals Histogram** - Shows distribution of prediction errors.

## 🌎 How to Use
### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/LSTM-N2O-Prediction.git
cd LSTM-N2O-Prediction
