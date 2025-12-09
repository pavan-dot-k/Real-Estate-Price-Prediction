import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Define custom layers (same as in training file)
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
    def build(self, input_shape):
        # Create positional encoding matrix
        position = np.arange(self.sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pos_encoding = np.zeros((self.sequence_length, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)
        
    def call(self, inputs):
        return inputs + self.pos_encoding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config

class GELU(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x):
        return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

class AttentionPooling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_dense = None
    
    def build(self, input_shape):
        self.attention_dense = Dense(1, activation='softmax')
        super().build(input_shape)
    
    def call(self, x):
        # x shape: (batch, seq_len, d_model)
        attention_weights = self.attention_dense(x)  # (batch, seq_len, 1)
        # Weighted sum using keras ops
        x_att = tf.reduce_sum(x * attention_weights, axis=1)  # (batch, d_model)
        return x_att

def load_trained_model():
    """Load the trained Transformer model and scalers"""
    print("Loading trained Transformer model and scalers...")
    
    # Define custom objects for loading the model
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'GELU': GELU,
        'AttentionPooling': AttentionPooling
    }
    
    # Load model with custom objects
    model = load_model('models/transformer_house_price_model.keras', custom_objects=custom_objects)
    
    # Load scalers
    with open('models/transformer_scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('models/transformer_scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    # Load training info
    with open('models/transformer_training_info.pkl', 'rb') as f:
        training_info = pickle.load(f)
    
    return model, scaler_X, scaler_y, training_info

def get_zipcode_data(zipcode, df):
    """Get historical data for a specific zipcode"""
    zip_data = df[df['zip'] == zipcode].copy()
    
    if len(zip_data) == 0:
        raise ValueError(f"No data found for zipcode {zipcode}")
    
    zip_data = zip_data.sort_values('date')
    return zip_data

def predict_future_prices(zipcode, n_months=12):
    """
    Predict house prices for the next n_months using Transformer model
    Args:
        zipcode: The zipcode to predict for
        n_months: Number of months to predict (default: 12)
    Returns:
        DataFrame with predictions
    """
    # Load model and scalers
    model, scaler_X, scaler_y, training_info = load_trained_model()
    
    sequence_length = training_info['sequence_length']
    feature_names = training_info['feature_names']
    
    # Load dataset
    print(f"\nLoading dataset for zipcode {zipcode}...")
    df = pd.read_csv('datasets/FInal Dataset/dataset_for_training.csv')
    
    # Preprocessing
    if 'month/year' in df.columns:
        df['year'] = pd.to_datetime(df['month/year'], format='%m/%Y').dt.year
        df['month'] = pd.to_datetime(df['month/year'], format='%m/%Y').dt.month
        df['date'] = pd.to_datetime(df['month/year'], format='%m/%Y')
    
    if 'ZHVI' in df.columns:
        df = df.rename(columns={'ZHVI': 'avg_house_price'})
    
    # Convert numeric columns
    numeric_columns = ['year', 'month', 'zip', 'median_income', 'crime_rate_per_1000', 'avg_house_price']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Get data for the specific zipcode
    zip_data = get_zipcode_data(zipcode, df)
    
    # Fill missing values
    zip_data['median_income'] = zip_data['median_income'].fillna(zip_data['median_income'].median())
    zip_data['crime_rate_per_1000'] = zip_data['crime_rate_per_1000'].fillna(zip_data['crime_rate_per_1000'].median())
    zip_data = zip_data.dropna(subset=['avg_house_price'])
    
    if len(zip_data) < sequence_length:
        raise ValueError(f"Not enough historical data for zipcode {zipcode}. Need at least {sequence_length} months.")
    
    print(f"Found {len(zip_data)} months of data for zipcode {zipcode}")
    print(f"Date range: {zip_data['date'].min()} to {zip_data['date'].max()}")
    print(f"Current Price: ${zip_data['avg_house_price'].iloc[-1]:,.2f}")
    
    # Check for NaN in the feature columns
    print(f"\nChecking for NaN in features:")
    for col in feature_names:
        nan_count = zip_data[col].isna().sum()
        if nan_count > 0:
            print(f"  {col}: {nan_count} NaN values")
    
    # Get the last sequence_length months of data
    last_sequence = zip_data[feature_names].values[-sequence_length:]
    
    print(f"\nLast sequence shape: {last_sequence.shape}")
    print(f"Last sequence has NaN: {np.isnan(last_sequence).any()}")
    if np.isnan(last_sequence).any():
        nan_positions = np.argwhere(np.isnan(last_sequence))
        print(f"NaN positions (row, col): {nan_positions[:5]}")  # Show first 5
        for row, col in nan_positions[:5]:
            print(f"  Row {row}, Feature '{feature_names[col]}': NaN")
    
    # Store last known values for features
    last_year = zip_data['year'].iloc[-1]
    last_month = zip_data['month'].iloc[-1]
    last_zip = zip_data['zip'].iloc[-1]
    last_median_income = zip_data['median_income'].iloc[-1]
    last_crime_rate = zip_data['crime_rate_per_1000'].iloc[-1]
    last_date = zip_data['date'].iloc[-1]
    
    # Make predictions
    predictions = []
    current_sequence = last_sequence.copy()
    
    print(f"\nGenerating predictions for the next {n_months} months...")
    
    current_year = last_year
    current_month = last_month
    
    for i in range(n_months):
        # Update month and year
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
        
        # Check for NaN in current sequence before scaling
        if np.isnan(current_sequence).any():
            print(f"Warning: NaN detected in sequence at iteration {i}")
            print(f"Sequence stats: min={np.nanmin(current_sequence)}, max={np.nanmax(current_sequence)}")
        
        # Scale the current sequence
        current_sequence_scaled = scaler_X.transform(current_sequence)
        current_sequence_scaled = current_sequence_scaled.reshape(1, sequence_length, len(feature_names))
        
        # Check for NaN after scaling
        if np.isnan(current_sequence_scaled).any():
            print(f"Warning: NaN detected in scaled sequence at iteration {i}")
        
        # Predict
        prediction_scaled = model.predict(current_sequence_scaled, verbose=0)
        prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
        
        # Check if prediction is valid
        if np.isnan(prediction):
            print(f"Warning: NaN prediction at iteration {i}")
            print(f"Prediction scaled: {prediction_scaled}")
            # Use last known price as fallback
            if i == 0:
                prediction = zip_data['avg_house_price'].iloc[-1]
            else:
                prediction = predictions[-1]
            print(f"Using fallback prediction: {prediction}")
        
        predictions.append(prediction)
        
        # Update sequence for next prediction
        # Assume features remain relatively constant (you can modify this logic)
        new_row = np.array([[current_year, current_month, last_zip, 
                           last_median_income, last_crime_rate]])
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    # Create results DataFrame
    future_dates = [last_date + timedelta(days=30*(i+1)) for i in range(n_months)]
    
    results_df = pd.DataFrame({
        'date': future_dates,
        'zipcode': zipcode,
        'predicted_price': predictions
    })
    
    results_df['month_year'] = results_df['date'].dt.strftime('%m/%Y')
    
    return results_df, zip_data

def plot_predictions(zip_data, predictions_df, zipcode):
    """Plot historical data and predictions"""
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    plt.plot(zip_data['date'], zip_data['avg_house_price'], 
             label='Historical Price', marker='o', linewidth=2, markersize=4)
    
    # Plot predictions
    plt.plot(predictions_df['date'], predictions_df['predicted_price'], 
             label='Predicted Price', marker='s', linewidth=2, linestyle='--', 
             color='red', markersize=4)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('House Price ($)', fontsize=12)
    plt.title(f'Transformer Model: House Price Prediction for Zipcode {zipcode}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'models/transformer_prediction_plot_{zipcode}.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: models/transformer_prediction_plot_{zipcode}.png")
    plt.show()

def main():
    """Main function to run predictions"""
    
    # Specify the zipcode you want to predict for
    # Change this to any zipcode in your dataset
    zipcode =85281  # Example zipcode - Change this to your desired zipcode
    n_months = 12    # Predict for the next 12 months
    
    try:
        # Make predictions
        predictions_df, historical_data = predict_future_prices(zipcode, n_months)
        
        # Display results
        print("\n" + "="*80)
        print(f"TRANSFORMER MODEL PREDICTIONS FOR ZIPCODE {zipcode}")
        print("="*80)
        print(predictions_df[['month_year', 'predicted_price']].to_string(index=False))
        
        # Calculate statistics
        current_price = historical_data['avg_house_price'].iloc[-1]
        predicted_price_12m = predictions_df['predicted_price'].iloc[-1]
        price_change = predicted_price_12m - current_price
        price_change_pct = (price_change / current_price) * 100
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Current Price (Latest):        ${current_price:,.2f}")
        print(f"Predicted Price (12 months):   ${predicted_price_12m:,.2f}")
        print(f"Expected Change:               ${price_change:,.2f} ({price_change_pct:+.2f}%)")
        print("="*80)
        
        # Save predictions to CSV
        output_file = f'models/transformer_predictions_{zipcode}.csv'
        predictions_df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")
        
        # Plot predictions
        plot_predictions(historical_data, predictions_df, zipcode)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nMake sure you have:")
        print("1. Trained the model first by running transformer_training.py")
        print("2. Specified a valid zipcode that exists in the dataset")

if __name__ == "__main__":
    main()
