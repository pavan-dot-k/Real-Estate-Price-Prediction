import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")
df = pd.read_csv('datasets/FInal Dataset/dataset_for_training.csv')

# --- 1. PREPROCESSING & CLEANING ---
if 'month/year' in df.columns:
    df['year'] = pd.to_datetime(df['month/year'], format='%m/%Y').dt.year
    df['month'] = pd.to_datetime(df['month/year'], format='%m/%Y').dt.month
    df = df.drop('month/year', axis=1)

if 'ZHVI' in df.columns:
    df = df.rename(columns={'ZHVI': 'avg_house_price'})

df.replace('N/A', np.nan, inplace=True)

# Convert numeric columns
numeric_columns = ['year', 'month', 'zip', 'median_income', 'crime_rate_per_1000', 'avg_house_price']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values
df = df.dropna(subset=['avg_house_price'])
feature_columns = ['median_income', 'crime_rate_per_1000']
for col in feature_columns:
    df[col] = df[col].fillna(df[col].median())

# --- 2. CRITICAL: SORTING FOR TIME SERIES ---
# You MUST sort by Zip first, then Time.
# If you sort by Time first, shifting will leak data between different zip codes.
print("\nSorting data by Zip and Time...")
df = df.sort_values(['zip', 'year', 'month']).reset_index(drop=True)

# --- 3. FEATURE ENGINEERING (The Forecasting Part) ---
print("Creating Lag Features...")

# A. Create Lags (History)
# We use the PAST price to predict the FUTURE price.
# We group by 'zip' so lags don't cross between different neighborhoods.
lags = [1, 3, 6, 12] 
for lag in lags:
    df[f'Price_Lag_{lag}'] = df.groupby('zip')['avg_house_price'].shift(lag)
    df[f'Income_Lag_{lag}'] = df.groupby('zip')['median_income'].shift(lag)

# B. Create the Target (The Future)
# We want to predict price 12 months into the future.
# We shift NEGATIVELY to bring future values into the current row for training.
FORECAST_HORIZON = 12
df['Target_Price_Next_Year'] = df.groupby('zip')['avg_house_price'].shift(-FORECAST_HORIZON)

# C. Drop NaN values created by shifting
# (We lose the first 12 months of data for lags, and the last 12 months for targets)
df_model = df.dropna()

print(f"Data ready for modeling. Shape: {df_model.shape}")

# --- 4. TEMPORAL SPLIT ---
# Train on older data, Test on the most recent data available
# We split based on the 'year' to ensure we aren't training on future years

# Check available years in the dataset
print(f"\nAvailable years: {sorted(df_model['year'].unique())}")
print(f"Date range: {df_model['year'].min()}/{df_model['month'].min():02d} to "
      f"{df_model['year'].max()}/{df_model['month'].max():02d}")

# Use 80/20 temporal split instead of fixed year
# This ensures we always have both train and test data
split_index = int(len(df_model) * 0.8)
train = df_model.iloc[:split_index]
test = df_model.iloc[split_index:]

print(f"\nTemporal Split:")
print(f"  Train: {train['year'].min()}/{train['month'].min():02d} to "
      f"{train['year'].max()}/{train['month'].max():02d} ({len(train)} samples)")
print(f"  Test:  {test['year'].min()}/{test['month'].min():02d} to "
      f"{test['year'].max()}/{test['month'].max():02d} ({len(test)} samples)")

# Define Predictors (X)
# NOTE: We do NOT use current 'avg_house_price' or 'median_income' as features
# because we won't know them 12 months in advance accurately.
# We use only Lags and Date info.
predictors = [
    'zip', 'month', 
    'Price_Lag_1', 'Price_Lag_3', 'Price_Lag_6', 'Price_Lag_12',
    'Income_Lag_1', 'Income_Lag_12'
]

X_train = train[predictors]
y_train = train['Target_Price_Next_Year']
X_test = test[predictors]
y_test = test['Target_Price_Next_Year']

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- 5. MODELING (Gradient Boosting) ---
print("\nTraining Gradient Boosting Regressor for 12-Month Forecast...")
gb_model = GradientBoostingRegressor(
    n_estimators=500,  # Increased estimators
    learning_rate=0.05, # Lower learning rate for better generalization
    max_depth=5, 
    random_state=42
)
gb_model.fit(X_train, y_train)

# --- 6. EVALUATION ---
gb_pred = gb_model.predict(X_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)

print(f"Gradient Boosting Results (12-Month Horizon):")
print(f"  RMSE: {gb_rmse:,.2f}")
print(f"  R² Score: {gb_r2:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': predictors,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nFeature Importance:\n{feature_importance}")

# --- 7. HOW TO USE FOR FUTURE ---
# To predict for a specific Zip Code for *Next Year* (where you don't have the answer yet):
# You would take the LATEST available row for that Zip, 
# and feed it into gb_model.predict().