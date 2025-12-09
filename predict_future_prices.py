import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Custom layer classes for Transformer model (must be defined before loading the model)
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
        super().build(input_shape)
        
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
        x_att = keras.ops.sum(x * attention_weights, axis=1)  # (batch, d_model)
        return x_att

# Constants
SEQUENCE_LENGTH = 12
FEATURE_COLS = ['year', 'month', 'zip', 'median_income', 'crime_rate_per_1000', 'business_count']
DATASET_PATH = 'datasets/FInal Dataset/dataset_for_training.csv'
LSTM_MODEL_PATH = 'lstm_house_price_model.keras'
TRANSFORMER_MODEL_PATH = 'transformer_house_price_model.keras'


def load_and_preprocess_data():
    """
    Load and preprocess the dataset (same logic as training scripts).
    Returns cleaned DataFrame.
    """
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Initial dataset shape: {df.shape}")
    
    # Preprocessing similar to training scripts
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
    
    # Drop rows with missing target values
    df = df.dropna(subset=['avg_house_price'])
    
    # Fill missing feature values with median
    feature_columns = ['median_income', 'crime_rate_per_1000', 'business_count']
    for col in feature_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Sort by zip, year, and month for proper sequence creation
    df = df.sort_values(['zip', 'year', 'month']).reset_index(drop=True)
    
    print(f"Dataset shape after cleaning: {df.shape}")
    return df


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


def recreate_scalers(df):
    """
    Recreate StandardScaler and MinMaxScaler from training data.
    Uses the same train/test split logic as training scripts.
    Returns scaler_X, scaler_y
    """
    print("Recreating scalers from training data...")
    
    # Group by zip code to create proper time series sequences
    zip_sequences = {}
    
    for zip_code in df['zip'].unique():
        zip_mask = df['zip'] == zip_code
        zip_df = df[zip_mask].copy()
        
        if len(zip_df) > SEQUENCE_LENGTH:
            zip_X = zip_df[FEATURE_COLS].values
            zip_y = zip_df['avg_house_price'].values
            
            X_seq, y_seq = create_sequences(zip_X, zip_y, SEQUENCE_LENGTH)
            zip_sequences[zip_code] = (X_seq, y_seq)
    
    # Split zip codes into train and test (80/20 split by zip codes) - same as training
    zip_codes = list(zip_sequences.keys())
    np.random.seed(42)
    np.random.shuffle(zip_codes)
    train_zip_size = int(0.8 * len(zip_codes))
    train_zips = zip_codes[:train_zip_size]
    
    # Combine sequences for train set only (to fit scalers)
    X_train_list, y_train_list = [], []
    
    for zip_code in train_zips:
        X_seq, y_seq = zip_sequences[zip_code]
        X_train_list.append(X_seq)
        y_train_list.append(y_seq)
    
    X_train = np.vstack(X_train_list)
    y_train = np.hstack(y_train_list)
    
    # Scale features and target (fit only on training data)
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()
    
    # Reshape for scaling: (samples * timesteps, features)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    
    scaler_X.fit(X_train_reshaped)
    scaler_y.fit(y_train.reshape(-1, 1))
    
    print("Scalers recreated successfully.")
    return scaler_X, scaler_y


def load_model(model_path, model_type):
    """
    Load a trained model from file.
    For Transformer models, includes custom objects for custom layers.
    """
    print(f"Loading {model_type} model from {model_path}...")
    try:
        # For Transformer model, provide custom objects
        if model_type.lower() == 'transformer':
            custom_objects = {
                'PositionalEncoding': PositionalEncoding,
                'GELU': GELU,
                'AttentionPooling': AttentionPooling
            }
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
        else:
            model = keras.models.load_model(model_path)
        
        print(f"{model_type} model loaded successfully.")
        print(f"Model input shape: {model.input_shape}")
        return model
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        raise


def train_regression_models(df):
    """
    Train Random Forest and Gradient Boosting models from the dataset.
    Uses the same preprocessing and train/test split as regression_analysis.py
    Returns: rf_model, gb_model
    """
    print("Training Random Forest and Gradient Boosting models...")
    
    # Prepare features and target (same as regression_analysis.py)
    X = df[FEATURE_COLS]
    y = df['avg_house_price']
    
    # Train/test split (same as regression_analysis.py)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Random Forest (no scaling needed)
    print("  Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Train Gradient Boosting (no scaling needed)
    print("  Training Gradient Boosting Regressor...")
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1)
    gb_model.fit(X_train, y_train)
    
    print("Regression models trained successfully.")
    return rf_model, gb_model


def get_zipcode_history(df, zipcode, n_months=12):
    """
    Extract the last N months of historical data for a specific zipcode.
    Returns DataFrame with historical data sorted by year and month.
    """
    zipcode = float(zipcode) if isinstance(zipcode, str) else zipcode
    
    zip_mask = df['zip'] == zipcode
    zip_df = df[zip_mask].copy()
    
    if len(zip_df) == 0:
        raise ValueError(f"Zipcode {zipcode} not found in dataset.")
    
    if len(zip_df) < n_months:
        raise ValueError(f"Zipcode {zipcode} has only {len(zip_df)} months of data, but {n_months} months are required.")
    
    # Sort by year and month to get the most recent data
    zip_df = zip_df.sort_values(['year', 'month']).reset_index(drop=True)
    
    # Get last n_months
    history = zip_df.tail(n_months).copy()
    
    print(f"Found {len(history)} months of historical data for zipcode {zipcode}")
    print(f"Date range: {int(history.iloc[0]['year'])}/{int(history.iloc[0]['month']):02d} to {int(history.iloc[-1]['year'])}/{int(history.iloc[-1]['month']):02d}")
    
    return history


def generate_future_features(last_sequence_entry, month, year, zipcode):
    """
    Generate feature values for a future month.
    Uses last known values for median_income, crime_rate_per_1000, and business_count.
    
    Parameters:
    - last_sequence_entry: numpy array of shape (6,) with last sequence features
    - month: target month (1-12)
    - year: target year
    - zipcode: target zipcode
    """
    # Extract feature indices from FEATURE_COLS
    # FEATURE_COLS = ['year', 'month', 'zip', 'median_income', 'crime_rate_per_1000', 'business_count']
    features = {
        'year': year,
        'month': month,
        'zip': zipcode,
        'median_income': last_sequence_entry[3],  # index 3
        'crime_rate_per_1000': last_sequence_entry[4],  # index 4
        'business_count': last_sequence_entry[5]  # index 5
    }
    
    # Convert to array in the same order as FEATURE_COLS
    feature_array = np.array([features[col] for col in FEATURE_COLS])
    return feature_array


def predict_next_month(sequence, model, scaler_X, scaler_y):
    """
    Predict the next month's price using a sequence of 12 months.
    sequence: numpy array of shape (12, 6) - 12 months of features
    Returns: predicted price (inverse transformed)
    """
    # Scale the sequence
    sequence_reshaped = sequence.reshape(-1, sequence.shape[1])
    sequence_scaled = scaler_X.transform(sequence_reshaped).reshape(sequence.shape)
    
    # Add batch dimension: (1, 12, 6)
    sequence_batch = np.expand_dims(sequence_scaled, axis=0)
    
    # Predict
    prediction_scaled = model.predict(sequence_batch, verbose=0)
    
    # Inverse transform
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
    
    return prediction


def predict_with_regression_model(feature_vector, model):
    """
    Predict price using Random Forest or Gradient Boosting model.
    These models don't use sequences - they predict directly from a feature vector.
    
    Parameters:
    - feature_vector: numpy array of shape (6,) with features [year, month, zip, median_income, crime_rate_per_1000, business_count]
    - model: trained RandomForestRegressor or GradientBoostingRegressor
    
    Returns: predicted price
    """
    # Reshape to 2D array for prediction (sklearn expects 2D)
    feature_vector_2d = feature_vector.reshape(1, -1)
    prediction = model.predict(feature_vector_2d)[0]
    return prediction


def predict_next_12_months(zipcode, model_type='both', future_features=None):
    """
    Predict the next 12 months of house prices for a specific zipcode.
    
    Parameters:
    - zipcode: Target zipcode (int or string)
    - model_type: 'lstm', 'transformer', 'random_forest', 'gradient_boosting', 'both', or 'all'
      - 'both': LSTM and Transformer
      - 'all': All models (LSTM, Transformer, Random Forest, Gradient Boosting)
    - future_features: Optional dict with future feature values (not implemented yet)
    
    Returns:
    - DataFrame with predictions for next 12 months
    """
    print("\n" + "="*60)
    print(f"PREDICTING NEXT 12 MONTHS FOR ZIPCODE {zipcode}")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Get historical data for the zipcode
    history = get_zipcode_history(df, zipcode, n_months=SEQUENCE_LENGTH)
    
    # Determine which models to use
    use_lstm = model_type in ['lstm', 'both', 'all']
    use_transformer = model_type in ['transformer', 'both', 'all']
    use_rf = model_type in ['random_forest', 'all']
    use_gb = model_type in ['gradient_boosting', 'all']
    
    # Load/train models
    sequence_models = {}  # LSTM and Transformer (sequence-based)
    regression_models = {}  # Random Forest and Gradient Boosting (non-sequence)
    
    if use_lstm or use_transformer:
        # Recreate scalers for sequence models
        scaler_X, scaler_y = recreate_scalers(df)
        
        if use_lstm:
            sequence_models['lstm'] = load_model(LSTM_MODEL_PATH, 'LSTM')
        if use_transformer:
            sequence_models['transformer'] = load_model(TRANSFORMER_MODEL_PATH, 'Transformer')
    else:
        scaler_X, scaler_y = None, None
    
    if use_rf or use_gb:
        rf_model, gb_model = train_regression_models(df)
        if use_rf:
            regression_models['random_forest'] = rf_model
        if use_gb:
            regression_models['gradient_boosting'] = gb_model
    
    # Prepare initial sequence from historical data (for sequence models)
    initial_sequence = history[FEATURE_COLS].values if sequence_models else None
    
    # Get the last row for future feature generation
    last_row = history.iloc[-1]
    last_year = int(last_row['year'])
    last_month = int(last_row['month'])
    last_feature_vector = history[FEATURE_COLS].iloc[-1].values  # For regression models
    
    # Initialize predictions dictionary
    predictions = {}
    
    # Predict using sequence-based models (LSTM, Transformer)
    for model_name, model in sequence_models.items():
        print(f"\nMaking predictions using {model_name.upper()} model...")
        
        # Start with historical sequence
        current_sequence = initial_sequence.copy()
        model_predictions = []
        
        # Predict 12 months ahead
        for i in range(12):
            # Calculate next month and year
            next_month = last_month + i + 1
            next_year = last_year
            
            # Handle year rollover
            while next_month > 12:
                next_month -= 12
                next_year += 1
            
            # Predict next month
            predicted_price = predict_next_month(current_sequence, model, scaler_X, scaler_y)
            model_predictions.append({
                'year': next_year,
                'month': next_month,
                'predicted_price': predicted_price
            })
            
            # Generate features for the predicted month (use last entry of current sequence)
            last_sequence_entry = current_sequence[-1]
            future_feature = generate_future_features(last_sequence_entry, next_month, next_year, zipcode)
            
            # Update sequence: remove first month, add predicted month
            current_sequence = np.vstack([current_sequence[1:], future_feature])
            
            print(f"  Month {i+1}/12: {next_year}/{next_month:02d} - ${predicted_price:,.2f}")
        
        predictions[model_name] = model_predictions
    
    # Predict using regression models (Random Forest, Gradient Boosting)
    for model_name, model in regression_models.items():
        print(f"\nMaking predictions using {model_name.upper()} model...")
        
        model_predictions = []
        current_feature_vector = last_feature_vector.copy()
        
        # Predict 12 months ahead
        for i in range(12):
            # Calculate next month and year
            next_month = last_month + i + 1
            next_year = last_year
            
            # Handle year rollover
            while next_month > 12:
                next_month -= 12
                next_year += 1
            
            # Update feature vector with new month/year
            current_feature_vector[0] = next_year  # year
            current_feature_vector[1] = next_month  # month
            current_feature_vector[2] = zipcode  # zip
            
            # Predict next month (regression models don't need sequences)
            predicted_price = predict_with_regression_model(current_feature_vector, model)
            model_predictions.append({
                'year': next_year,
                'month': next_month,
                'predicted_price': predicted_price
            })
            
            print(f"  Month {i+1}/12: {next_year}/{next_month:02d} - ${predicted_price:,.2f}")
        
        predictions[model_name] = model_predictions
    
    # Create output DataFrame
    num_models = len(predictions)
    
    if num_models == 0:
        raise ValueError("No models selected for prediction.")
    
    if num_models == 1:
        # Single model predictions
        model_name = list(predictions.keys())[0]
        results = predictions[model_name]
        results_df = pd.DataFrame(results)
        results_df = results_df.rename(columns={'predicted_price': f'{model_name}_prediction'})
    else:
        # Multiple models - combine predictions
        results = []
        model_names = list(predictions.keys())
        
        for i in range(12):
            row = {
                'year': predictions[model_names[0]][i]['year'],
                'month': predictions[model_names[0]][i]['month']
            }
            
            # Add predictions from each model
            pred_values = []
            for model_name in model_names:
                pred_value = predictions[model_name][i]['predicted_price']
                row[f'{model_name}_prediction'] = pred_value
                pred_values.append(pred_value)
            
            # Calculate average if multiple models
            if len(pred_values) > 1:
                row['average_prediction'] = np.mean(pred_values)
            
            results.append(row)
        
        results_df = pd.DataFrame(results)
    
    return results_df, history


def display_predictions(predictions_df, history_df, zipcode):
    """
    Display predictions in a readable format.
    """
    print("\n" + "="*60)
    print(f"PREDICTION RESULTS FOR ZIPCODE {zipcode}")
    print("="*60)
    
    # Display historical context
    print("\nHistorical Data (Last 3 months):")
    print("-" * 60)
    hist_display = history_df[['year', 'month', 'avg_house_price']].tail(3)
    for _, row in hist_display.iterrows():
        print(f"  {int(row['year'])}/{int(row['month']):02d}: ${row['avg_house_price']:,.2f}")
    
    # Display predictions
    print("\nFuture Predictions (Next 12 months):")
    print("-" * 60)
    
    # Get all prediction columns
    pred_cols = [col for col in predictions_df.columns if 'prediction' in col]
    
    if 'average_prediction' in predictions_df.columns:
        # Multiple models with average
        for _, row in predictions_df.iterrows():
            pred_str = ", ".join([f"{col.replace('_prediction', '').title()}=${row[col]:,.2f}" 
                                 for col in pred_cols if col != 'average_prediction'])
            print(f"  {int(row['year'])}/{int(row['month']):02d}: {pred_str}, "
                  f"Average=${row['average_prediction']:,.2f}")
    elif len(pred_cols) > 1:
        # Multiple models without average
        for _, row in predictions_df.iterrows():
            pred_str = ", ".join([f"{col.replace('_prediction', '').title()}=${row[col]:,.2f}" 
                                 for col in pred_cols])
            print(f"  {int(row['year'])}/{int(row['month']):02d}: {pred_str}")
    else:
        # Single model
        pred_col = pred_cols[0]
        model_name = pred_col.replace('_prediction', '').title()
        for _, row in predictions_df.iterrows():
            print(f"  {int(row['year'])}/{int(row['month']):02d}: "
                  f"{model_name}=${row[pred_col]:,.2f}")
    
    print("\n" + "="*60)


def save_predictions(predictions_df, zipcode, output_path=None):
    """
    Save predictions to CSV file.
    """
    if output_path is None:
        output_path = f'predictions_zipcode_{zipcode}.csv'
    
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Get zipcode from command line or use default
    if len(sys.argv) > 1:
        zipcode = sys.argv[1]
    else:
        zipcode = 85001  # Default zipcode
    
    # Get model type from command line or use default
    if len(sys.argv) > 2:
        model_type = sys.argv[2].lower()
        valid_types = ['lstm', 'transformer', 'random_forest', 'gradient_boosting', 'both', 'all']
        if model_type not in valid_types:
            print(f"Invalid model type. Valid options: {', '.join(valid_types)}")
            print("Using 'both' (LSTM and Transformer) as default.")
            model_type = 'both'
    else:
        model_type = 'both'
    
    try:
        # Make predictions
        predictions_df, history_df = predict_next_12_months(zipcode, model_type=model_type)
        
        # Display results
        display_predictions(predictions_df, history_df, zipcode)
        
        # Save to CSV
        save_predictions(predictions_df, zipcode)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

