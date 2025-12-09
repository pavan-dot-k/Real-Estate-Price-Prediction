import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('datasets/FInal Dataset/dataset_for_training.csv')
print(f"Initial dataset shape: {df.shape}")

# Preprocessing similar to regression_analysis.py
if 'month/year' in df.columns:
    df['year'] = pd.to_datetime(df['month/year'], format='%m/%Y').dt.year
    df['month'] = pd.to_datetime(df['month/year'], format='%m/%Y').dt.month
    df = df.drop('month/year', axis=1)

if 'ZHVI' in df.columns:
    df = df.rename(columns={'ZHVI': 'avg_house_price'})

df.replace('N/A', np.nan, inplace=True)

numeric_columns = ['year', 'month', 'zip', 'median_income', 'crime_rate_per_1000', 'business_count', 'avg_house_price']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\nMissing values before cleaning:")
print(df.isnull().sum())
df = df.dropna(subset=['avg_house_price'])
feature_columns = ['median_income', 'crime_rate_per_1000', 'business_count']
for col in feature_columns:
    df[col] = df[col].fillna(df[col].median())

print(f"\nDataset shape after cleaning: {df.shape}")
print(f"\nMissing values after cleaning:")
print(df.isnull().sum())

# Sort by zip, year, and month for proper sequence creation
df = df.sort_values(['zip', 'year', 'month']).reset_index(drop=True)

# Prepare features and target
feature_cols = ['year', 'month', 'zip', 'median_income', 'crime_rate_per_1000', 'business_count']

# Function to create sequences for LSTM
def create_sequences(X, y, sequence_length=12):
    """
    Create sequences from the data.
    Each sequence contains 'sequence_length' time steps to predict the next value.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)

# Create sequences (using 12 months as sequence length)
sequence_length = 12
print(f"\nCreating sequences with length: {sequence_length}")

# Group by zip code to create proper time series sequences
# Store sequences per zip code for proper train/test split
zip_sequences = {}

for zip_code in df['zip'].unique():
    zip_mask = df['zip'] == zip_code
    zip_df = df[zip_mask].copy()
    
    if len(zip_df) > sequence_length:
        zip_X = zip_df[feature_cols].values
        zip_y = zip_df['avg_house_price'].values
        
        X_seq, y_seq = create_sequences(zip_X, zip_y, sequence_length)
        zip_sequences[zip_code] = (X_seq, y_seq)

# Split zip codes into train and test (80/20 split by zip codes)
zip_codes = list(zip_sequences.keys())
np.random.seed(42)
np.random.shuffle(zip_codes)
train_zip_size = int(0.8 * len(zip_codes))
train_zips = zip_codes[:train_zip_size]
test_zips = zip_codes[train_zip_size:]

print(f"\nTrain zip codes: {len(train_zips)}, Test zip codes: {len(test_zips)}")

# Combine sequences for train and test sets
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for zip_code in train_zips:
    X_seq, y_seq = zip_sequences[zip_code]
    X_train_list.append(X_seq)
    y_train_list.append(y_seq)

for zip_code in test_zips:
    X_seq, y_seq = zip_sequences[zip_code]
    X_test_list.append(X_seq)
    y_test_list.append(y_seq)

X_train = np.vstack(X_train_list)
y_train = np.hstack(y_train_list)
X_test = np.vstack(X_test_list)
y_test = np.hstack(y_test_list)

print(f"\nSequences shape - X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Sequences shape - X_test: {X_test.shape}, y_test: {y_test.shape}")

# Scale features and target (fit only on training data to avoid data leakage)
scaler_X = StandardScaler()
scaler_y = MinMaxScaler()

# Reshape for scaling: (samples * timesteps, features)
X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
X_test_reshaped = X_test.reshape(-1, X_test.shape[2])

X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Sequence shape: {X_train.shape[1:]}")

# Build LSTM Model
print("\n" + "="*60)
print("Building LSTM Model...")
print("="*60)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    Dropout(0.3),
    LSTM(32, return_sequences=True),
    Dropout(0.3),
    LSTM(16, return_sequences=False),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='mse',
    metrics=['mae']
)

print("\nModel Architecture:")
model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1,
    min_delta=0.0001
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=0.00001,
    verbose=1
)

# Train the model
print("\n" + "="*60)
print("Training LSTM Model...")
print("="*60)

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Make predictions
print("\n" + "="*60)
print("Making Predictions...")
print("="*60)

y_train_pred_scaled = model.predict(X_train_scaled, verbose=0)
y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)

# Inverse transform predictions and actual values
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
y_train_actual = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# Calculate metrics
train_mse = mean_squared_error(y_train_actual, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train_actual, y_train_pred)
train_r2 = r2_score(y_train_actual, y_train_pred)

test_mse = mean_squared_error(y_test_actual, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test_actual, y_test_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)

# Print results
print("\n" + "="*60)
print("LSTM MODEL RESULTS")
print("="*60)

print("\nTraining Set Results:")
print(f"  MSE:  {train_mse:,.2f}")
print(f"  RMSE: {train_rmse:,.2f}")
print(f"  MAE:  {train_mae:,.2f}")
print(f"  R² Score: {train_r2:.4f}")

print("\nTest Set Results:")
print(f"  MSE:  {test_mse:,.2f}")
print(f"  RMSE: {test_rmse:,.2f}")
print(f"  MAE:  {test_mae:,.2f}")
print(f"  R² Score: {test_r2:.4f}")

# Save the model
model.save('lstm_house_price_model.keras')
print("\nModel saved as 'lstm_house_price_model.keras'")

print("\n" + "="*60)
print("Training completed successfully!")
print("="*60)

