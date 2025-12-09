import pandas as pd
import numpy as np
import joblib
from datetime import datetime

print("Loading pre-trained model...")
try:
    # Load the saved model
    model = joblib.load('gb_price_predictor.pkl')
    
    # Load model metadata
    try:
        metadata = joblib.load('model_metadata.pkl')
        model_name = metadata['model_name']
        requires_scaling = metadata['requires_scaling']
        print(f"✅ Model loaded successfully: {model_name}")
        print(f"   R² Score: {metadata['r2_score']:.4f}")
        print(f"   RMSE: ${metadata['rmse']:,.2f}")
        print(f"   MAE: ${metadata['mae']:,.2f}")
    except FileNotFoundError:
        # Fallback if no metadata (backward compatibility)
        model_name = "Gradient Boosting"
        requires_scaling = False
        print("✅ Model loaded successfully!")
    
    # Load scaler if needed
    if requires_scaling:
        scaler = joblib.load('scaler.pkl')
        print(f"✅ Scaler loaded (required for {model_name})")
    else:
        scaler = None
        
except FileNotFoundError:
    print("❌ Model file not found! Please run 'regression.py' first to train and save the model.")
    exit()

# Load the dataset to get the latest data
print("\nLoading dataset...")
df = pd.read_csv('datasets/FInal Dataset/dataset_for_training.csv')

# --- SAME PREPROCESSING AS TRAINING ---
if 'month/year' in df.columns:
    df['year'] = pd.to_datetime(df['month/year'], format='%m/%Y').dt.year
    df['month'] = pd.to_datetime(df['month/year'], format='%m/%Y').dt.month
    df = df.drop('month/year', axis=1)

if 'ZHVI' in df.columns:
    df = df.rename(columns={'ZHVI': 'avg_house_price'})

df.replace('N/A', np.nan, inplace=True)

numeric_columns = ['year', 'month', 'zip', 'median_income', 'crime_rate_per_1000', 'avg_house_price']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['avg_house_price'])
feature_columns = ['median_income', 'crime_rate_per_1000']
for col in feature_columns:
    df[col] = df[col].fillna(df[col].median())

df = df.sort_values(['zip', 'year', 'month']).reset_index(drop=True)

# Create lag features (same as training)
lags = [1, 3, 6, 12]
for lag in lags:
    df[f'Price_Lag_{lag}'] = df.groupby('zip')['avg_house_price'].shift(lag)
    df[f'Income_Lag_{lag}'] = df.groupby('zip')['median_income'].shift(lag)

# Remove rows with NaN lag features
df_ready = df.dropna(subset=[f'Price_Lag_{lag}' for lag in lags])

# Predictors (must match training)
# NOTE: 'zip' is REMOVED because it causes issues with linear models
# (they treat zip codes as numeric: 90210 > 10001 is meaningless)
# The lag features already capture zip-specific history!
predictors = [
    'month', 
    'Price_Lag_1', 'Price_Lag_3', 'Price_Lag_6', 'Price_Lag_12',
    'Income_Lag_1', 'Income_Lag_12'
]

print(f"✅ Data prepared. Ready for predictions!")

# --- INTERACTIVE PREDICTION ---
print("\n" + "="*60)
print("PREDICT FUTURE HOUSE PRICES (Using Pre-Trained Model)")
print("="*60)

available_zips = sorted(df_ready['zip'].unique())
print(f"\n📍 Available Zip Codes ({len(available_zips)} total):")
print(available_zips)

while True:
    print("\n" + "-"*60)
    user_input = input("\nEnter a Zip Code (or 'quit' to exit): ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("👋 Goodbye!")
        break
    
    try:
        user_zip = int(user_input)
        
        if user_zip not in available_zips:
            print(f"❌ Zip code {user_zip} not found in dataset!")
            print(f"Available zip codes: {available_zips}")
            continue
        
        # Get the latest data for this zip code
        zip_data = df_ready[df_ready['zip'] == user_zip].copy()
        latest_row = zip_data.iloc[-1:].copy()
        
        # Get historical price trend
        recent_prices = zip_data.tail(12)[['year', 'month', 'avg_house_price']]
        
        print(f"\n{'='*60}")
        print(f"PREDICTION FOR ZIP CODE: {user_zip}")
        print(f"{'='*60}")
        
        # Display current information
        current_year = int(latest_row['year'].values[0])
        current_month = int(latest_row['month'].values[0])
        current_price = latest_row['avg_house_price'].values[0]
        current_income = latest_row['median_income'].values[0]
        
        print(f"\n📍 Latest Data Point:")
        print(f"   Date: {current_year}/{current_month:02d}")
        print(f"   Current Avg House Price: ${current_price:,.2f}")
        print(f"   Median Income: ${current_income:,.2f}")
        
        # Show recent price trend
        print(f"\n📊 Recent Price History (Last {len(recent_prices)} Months):")
        for _, row in recent_prices.iterrows():
            print(f"   {int(row['year'])}/{int(row['month']):02d}: ${row['avg_house_price']:,.2f}")
        
        # Calculate price change over last 12 months
        if len(recent_prices) >= 2:
            oldest_price = recent_prices.iloc[0]['avg_house_price']
            recent_change = current_price - oldest_price
            recent_change_pct = (recent_change / oldest_price) * 100
            print(f"\n   📈 Change over period: ${recent_change:,.2f} ({recent_change_pct:+.2f}%)")
        
        # Make prediction using the loaded model
        if requires_scaling and scaler is not None:
            # Scale the features if needed
            X_pred_scaled = scaler.transform(latest_row[predictors])
            predicted_price = model.predict(X_pred_scaled)[0]
        else:
            predicted_price = model.predict(latest_row[predictors])[0]
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Calculate prediction date (12 months ahead)
        pred_month = current_month
        pred_year = current_year + 1
        
        print(f"\n🔮 PREDICTION (12 Months Ahead):")
        print(f"   Predicted For: {pred_year}/{pred_month:02d}")
        print(f"   Predicted Avg House Price: ${predicted_price:,.2f}")
        print(f"   Expected Change: ${price_change:,.2f} ({price_change_pct:+.2f}%)")
        
        if price_change > 0:
            print(f"   📈 Market Trend: INCREASING")
        else:
            print(f"   📉 Market Trend: DECREASING")
        
        # Calculate monthly change
        monthly_change = price_change / 12
        print(f"   Average Monthly Change: ${monthly_change:,.2f}")
        
        # Investment insight
        print(f"\n💡 Investment Insight:")
        if price_change_pct > 10:
            print(f"   🔥 Strong growth expected! ({price_change_pct:.1f}%)")
        elif price_change_pct > 5:
            print(f"   ✅ Moderate growth expected ({price_change_pct:.1f}%)")
        elif price_change_pct > 0:
            print(f"   ⚠️  Slow growth expected ({price_change_pct:.1f}%)")
        else:
            print(f"   ⚠️  Price decline expected ({price_change_pct:.1f}%)")
        
        print(f"\n{'='*60}")
        
    except ValueError:
        print("❌ Invalid input! Please enter a numeric zip code.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

print("\n✨ Prediction session ended.")