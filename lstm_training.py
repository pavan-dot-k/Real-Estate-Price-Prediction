import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os
import matplotlib.pyplot as plt

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('datasets/FInal Dataset/dataset_for_training.csv')

# Convert month/year to datetime
df['date'] = pd.to_datetime(df['month/year'], format='%m/%Y')

# Drop rows with missing ZHVI values
df = df.dropna(subset=['ZHVI'])

# Fill missing values in other columns with median/mean
df['median_income'] = df['median_income'].fillna(df['median_income'].median())
df['crime_rate_per_1000'] = df['crime_rate_per_1000'].fillna(df['crime_rate_per_1000'].median())
df['business_count'] = df['business_count'].fillna(df['business_count'].median())

# Sort by zipcode and date
df = df.sort_values(['zip', 'date'])

print(f"Dataset shape: {df.shape}")
print(f"Number of unique zipcodes: {df['zip'].nunique()}")

# Prepare sequences for LSTM
def create_sequences(data, n_steps):
    """
    Create sequences for LSTM training with date tracking
    Args:
        data: DataFrame with features
        n_steps: Number of time steps to look back
    Returns:
        X: Input sequences
        y: Target values
        dates: Corresponding dates for each sequence (for time-based splitting)
    """
    X, y, dates = [], [], []
    
    for zip_code in data['zip'].unique():
        zip_data = data[data['zip'] == zip_code].copy()
        
        if len(zip_data) < n_steps + 1:
            continue
            
        # Select features
        features = zip_data[['median_income', 'crime_rate_per_1000', 'business_count', 'ZHVI']].values
        date_values = zip_data['date'].values
        
        for i in range(len(features) - n_steps):
            X.append(features[i:i+n_steps])
            y.append(features[i+n_steps, -1])  # ZHVI is the last column
            dates.append(date_values[i+n_steps])  # Date of the target value
    
    return np.array(X), np.array(y), np.array(dates)

# Parameters
n_steps = 12  # Use 12 months of data to predict the next month

print("\nCreating sequences...")
X, y, dates = create_sequences(df, n_steps)

print(f"Sequences created: {X.shape}")
print(f"Target shape: {y.shape}")

# TIME-BASED SPLIT (No look-ahead bias!)
# Sort by date to ensure chronological order
sort_idx = np.argsort(dates)
X = X[sort_idx]
y = y[sort_idx]
dates = dates[sort_idx]

# Split: Use 80% earliest data for training, 20% most recent for testing
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_train, dates_test = dates[:split_idx], dates[split_idx:]

print(f"\nTime-based split:")
print(f"  Training period: {pd.to_datetime(dates_train[0])} to {pd.to_datetime(dates_train[-1])}")
print(f"  Testing period:  {pd.to_datetime(dates_test[0])} to {pd.to_datetime(dates_test[-1])}")
print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Reshape for scaling
n_samples_train, n_steps_train, n_features = X_train.shape
X_train_reshaped = X_train.reshape(-1, n_features)
X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
X_train_scaled = X_train_scaled.reshape(n_samples_train, n_steps_train, n_features)

n_samples_test = X_test.shape[0]
X_test_reshaped = X_test.reshape(-1, n_features)
X_test_scaled = scaler_X.transform(X_test_reshaped)
X_test_scaled = X_test_scaled.reshape(n_samples_test, n_steps_train, n_features)

# Scale target
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Save scalers
os.makedirs('models', exist_ok=True)
with open('models/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('models/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("\nScalers saved successfully!")

# Build LSTM model
print("\nBuilding LSTM model...")
model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)),
    Dropout(0.2),
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(model.summary())

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('models/lstm_model.h5', save_best_only=True, monitor='val_loss')

# Train model
print("\nTraining model...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_scaled),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Evaluate model
print("\nEvaluating model...")
train_loss, train_mae = model.evaluate(X_train_scaled, y_train_scaled, verbose=0)
test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)

# Calculate R² score
y_train_pred_scaled = model.predict(X_train_scaled, verbose=0)
y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)

# Inverse transform to get actual values
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)  # Fixed: use predictions, not actuals
y_train_actual = scaler_y.inverse_transform(y_train_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

train_r2 = r2_score(y_train_actual, y_train_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)

print(f"\n{'='*60}")
print(f"MODEL EVALUATION METRICS")
print(f"{'='*60}")
print(f"\nTraining Metrics:")
print(f"  Loss (MSE):  {train_loss:.6f}")
print(f"  MAE:         {train_mae:.6f}")
print(f"  R² Score:    {train_r2:.4f}")
print(f"\nTest Metrics:")
print(f"  Loss (MSE):  {test_loss:.6f}")
print(f"  MAE:         {test_mae:.6f}")
print(f"  R² Score:    {test_r2:.4f}")
print(f"{'='*60}")

# Save the complete model
model.save('models/lstm_model_final.keras')

# Save training data info for prediction
training_info = {
    'n_steps': n_steps,
    'n_features': n_features,
    'feature_names': ['median_income', 'crime_rate_per_1000', 'business_count', 'ZHVI']
}

with open('models/training_info.pkl', 'wb') as f:
    pickle.dump(training_info, f)

print("\n" + "="*60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nFiles saved:")
print("  - models/lstm_model_final.keras")
print("  - models/lstm_model.h5")
print("  - models/scaler_X.pkl")
print("  - models/scaler_y.pkl")
print("  - models/training_info.pkl")
print("\nYou can now run lstm_prediction.py to make predictions!")
print("="*60)

# Plot actual vs predicted for visualization
print("\nGenerating validation plot...")

# Plot a slice of test data (first 100 samples)
subset_size = min(100, len(y_test_actual))

plt.figure(figsize=(14, 6))
plt.plot(dates_test[:subset_size], y_test_actual[:subset_size], label='Actual Price', marker='o', markersize=4)
plt.plot(dates_test[:subset_size], y_test_pred[:subset_size], label='Predicted Price', linestyle='--', marker='x', markersize=4)

plt.title(f'Actual vs Predicted Prices (Test Set - First {subset_size} samples)')
plt.xlabel('Date')
plt.ylabel('ZHVI Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('models/test_set_validation_plot.png', dpi=150)
print("✓ Validation plot saved to: models/test_set_validation_plot.png")
plt.close()