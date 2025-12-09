import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_trained_model():
    """Load the trained LSTM model and scalers"""
    print("Loading trained model and scalers...")
    
    # Load model
    model = load_model('models/lstm_model_final.keras')
    
    # Load scalers
    with open('models/scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('models/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    # Load training info
    with open('models/training_info.pkl', 'rb') as f:
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
    Predict house prices for the next n_months
    Args:
        zipcode: The zipcode to predict for
        n_months: Number of months to predict (default: 12)
    Returns:
        DataFrame with predictions
    """
    # Load model and scalers
    model, scaler_X, scaler_y, training_info = load_trained_model()
    
    n_steps = training_info['n_steps']
    feature_names = training_info['feature_names']
    
    # Load dataset
    print(f"\nLoading dataset for zipcode {zipcode}...")
    df = pd.read_csv('datasets/FInal Dataset/dataset_for_training.csv')
    df['date'] = pd.to_datetime(df['month/year'], format='%m/%Y')
    
    # Get data for the specific zipcode
    zip_data = get_zipcode_data(zipcode, df)
    
    # Fill missing values
    zip_data['median_income'] = zip_data['median_income'].fillna(zip_data['median_income'].median())
    zip_data['crime_rate_per_1000'] = zip_data['crime_rate_per_1000'].fillna(zip_data['crime_rate_per_1000'].median())
    zip_data['business_count'] = zip_data['business_count'].fillna(zip_data['business_count'].median())
    zip_data = zip_data.dropna(subset=['ZHVI'])
    
    if len(zip_data) < n_steps:
        raise ValueError(f"Not enough historical data for zipcode {zipcode}. Need at least {n_steps} months.")
    
    print(f"Found {len(zip_data)} months of data for zipcode {zipcode}")
    print(f"Date range: {zip_data['date'].min()} to {zip_data['date'].max()}")
    print(f"Current ZHVI: ${zip_data['ZHVI'].iloc[-1]:,.2f}")
    
    # Get the last n_steps months of data
    last_sequence = zip_data[['median_income', 'crime_rate_per_1000', 'business_count', 'ZHVI']].values[-n_steps:]
    
    # Store last known values for features that don't change much
    last_median_income = zip_data['median_income'].iloc[-1]
    last_crime_rate = zip_data['crime_rate_per_1000'].iloc[-1]
    last_business_count = zip_data['business_count'].iloc[-1]
    last_date = zip_data['date'].iloc[-1]
    
    # Make predictions
    predictions = []
    current_sequence = last_sequence.copy()
    
    print(f"\nGenerating predictions for the next {n_months} months...")
    
    for i in range(n_months):
        # Scale the current sequence
        current_sequence_scaled = scaler_X.transform(current_sequence)
        current_sequence_scaled = current_sequence_scaled.reshape(1, n_steps, len(feature_names))
        
        # Predict
        prediction_scaled = model.predict(current_sequence_scaled, verbose=0)
        prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
        
        predictions.append(prediction)
        
        # Update sequence for next prediction
        # Assume features remain relatively constant (you can modify this logic)
        new_row = np.array([last_median_income, last_crime_rate, last_business_count, prediction])
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    # Create results DataFrame
    future_dates = [last_date + timedelta(days=30*(i+1)) for i in range(n_months)]
    
    results_df = pd.DataFrame({
        'date': future_dates,
        'zipcode': zipcode,
        'predicted_ZHVI': predictions
    })
    
    results_df['month_year'] = results_df['date'].dt.strftime('%m/%Y')
    
    return results_df, zip_data

def plot_predictions(zip_data, predictions_df, zipcode):
    """Plot historical data and predictions"""
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    plt.plot(zip_data['date'], zip_data['ZHVI'], label='Historical ZHVI', marker='o', linewidth=2)
    
    # Plot predictions
    plt.plot(predictions_df['date'], predictions_df['predicted_ZHVI'], 
             label='Predicted ZHVI', marker='s', linewidth=2, linestyle='--', color='red')
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('House Price (ZHVI)', fontsize=12)
    plt.title(f'House Price Prediction for Zipcode {zipcode}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'models/prediction_plot_{zipcode}.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: models/prediction_plot_{zipcode}.png")
    plt.show()

def main():
    """Main function to run predictions"""
    
    # Specify the zipcode you want to predict for
    # Change this to any zipcode in your dataset
    zipcode = 85614  # Example zipcode
    n_months = 12    # Predict for the next 12 months
    
    try:
        # Make predictions
        predictions_df, historical_data = predict_future_prices(zipcode, n_months)
        
        # Display results
        print("\n" + "="*80)
        print(f"PREDICTIONS FOR ZIPCODE {zipcode}")
        print("="*80)
        print(predictions_df[['month_year', 'predicted_ZHVI']].to_string(index=False))
        
        # Calculate statistics
        current_price = historical_data['ZHVI'].iloc[-1]
        predicted_price_12m = predictions_df['predicted_ZHVI'].iloc[-1]
        price_change = predicted_price_12m - current_price
        price_change_pct = (price_change / current_price) * 100
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Current ZHVI (Latest):        ${current_price:,.2f}")
        print(f"Predicted ZHVI (12 months):   ${predicted_price_12m:,.2f}")
        print(f"Expected Change:              ${price_change:,.2f} ({price_change_pct:+.2f}%)")
        print("="*80)
        
        # Save predictions to CSV
        output_file = f'models/predictions_{zipcode}.csv'
        predictions_df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")
        
        # Plot predictions
        plot_predictions(historical_data, predictions_df, zipcode)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nMake sure you have:")
        print("1. Trained the model first by running lstm_training.py")
        print("2. Specified a valid zipcode that exists in the dataset")

if __name__ == "__main__":
    main()

