import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

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

# FIX 1: Zip-Specific Imputation
# Fill missing income/crime with the median of THAT zip code first.
feature_columns = ['median_income', 'crime_rate_per_1000']
for col in feature_columns:
    df[col] = df.groupby('zip')[col].transform(lambda x: x.fillna(x.median()))
    # If a zip has NO data at all, fall back to global median
    df[col] = df[col].fillna(df[col].median())

# --- 2. CRITICAL: SORTING FOR TIME SERIES ---
print("\nSorting data by Zip and Time...")
df = df.sort_values(['zip', 'year', 'month']).reset_index(drop=True)

# --- 3. FEATURE ENGINEERING ---
print("Creating Lag Features...")

# A. Create Lags (History)
lags = [1, 3, 6, 12] 
for lag in lags:
    df[f'Price_Lag_{lag}'] = df.groupby('zip')['avg_house_price'].shift(lag)
    df[f'Income_Lag_{lag}'] = df.groupby('zip')['median_income'].shift(lag)

# B. Create the Target (The Future)
FORECAST_HORIZON = 12
df['Target_Price_Next_Year'] = df.groupby('zip')['avg_house_price'].shift(-FORECAST_HORIZON)

# C. Drop NaN values
df_model = df.dropna()
print(f"Data ready for modeling. Shape: {df_model.shape}")

# --- 4. TEMPORAL SPLIT (CORRECTED) ---
# FIX 2: Split by DATE, not by Index
# This ensures we don't leak future data or split zip codes incorrectly.

# Get unique dates and find the 80% cutoff mark
unique_dates = df_model[['year', 'month']].drop_duplicates().sort_values(['year', 'month'])
split_idx = int(len(unique_dates) * 0.8)
cutoff_row = unique_dates.iloc[split_idx]
cutoff_year, cutoff_month = cutoff_row['year'], cutoff_row['month']

print(f"\nTemporal Split Cutoff: {cutoff_year}/{cutoff_month:02d}")

# Create masks for Train (Before cutoff) and Test (After cutoff)
train_mask = (df_model['year'] < cutoff_year) | ((df_model['year'] == cutoff_year) & (df_model['month'] < cutoff_month))
test_mask = ~train_mask

train = df_model[train_mask]
test = df_model[test_mask]

print(f"  Train: {len(train)} samples")
print(f"  Test:  {len(test)} samples")

# Define Predictors (X)
# FIX 3: Removed 'zip' from predictors
# We rely on the Lags to capture specific neighborhood price history.
predictors = [
    'month', 
    'Price_Lag_1', 'Price_Lag_3', 'Price_Lag_6', 'Price_Lag_12',
    'Income_Lag_1', 'Income_Lag_12'
]

X_train = train[predictors]
y_train = train['Target_Price_Next_Year']
X_test = test[predictors]
y_test = test['Target_Price_Next_Year']

# --- 5. SCALE DATA FOR LINEAR MODELS ---
print("\nScaling features for linear models...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. TRAIN MULTIPLE MODELS ---
print("\n" + "="*60)
print("TRAINING MULTIPLE MODELS FOR 12-MONTH FORECAST")
print("="*60)

results = {}

# Model 1: Gradient Boosting
print("\n1. Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)
gb_mae = mean_absolute_error(y_test, gb_pred)

results['Gradient Boosting'] = {'model': gb_model, 'predictions': gb_pred, 'RMSE': gb_rmse, 'MAE': gb_mae, 'R2': gb_r2, 'scaled': False}
print(f"   RMSE: {gb_rmse:,.2f} | R²: {gb_r2:.4f}")

# Model 2: Random Forest
print("\n2. Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

results['Random Forest'] = {'model': rf_model, 'predictions': rf_pred, 'RMSE': rf_rmse, 'MAE': rf_mae, 'R2': rf_r2, 'scaled': False}
print(f"   RMSE: {rf_rmse:,.2f} | R²: {rf_r2:.4f}")

# Model 3: Lasso
print("\n3. Training Lasso Regression...")
lasso_model = Lasso(alpha=1.0, random_state=42, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)
lasso_pred = lasso_model.predict(X_test_scaled)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_r2 = r2_score(y_test, lasso_pred)
lasso_mae = mean_absolute_error(y_test, lasso_pred)

results['Lasso'] = {'model': lasso_model, 'predictions': lasso_pred, 'RMSE': lasso_rmse, 'MAE': lasso_mae, 'R2': lasso_r2, 'scaled': True}
print(f"   RMSE: {lasso_rmse:,.2f} | R²: {lasso_r2:.4f}")

# Model 4: Ridge
print("\n4. Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0, random_state=42, max_iter=10000)
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_r2 = r2_score(y_test, ridge_pred)
ridge_mae = mean_absolute_error(y_test, ridge_pred)

results['Ridge'] = {'model': ridge_model, 'predictions': ridge_pred, 'RMSE': ridge_rmse, 'MAE': ridge_mae, 'R2': ridge_r2, 'scaled': True}
print(f"   RMSE: {ridge_rmse:,.2f} | R²: {ridge_r2:.4f}")

# Model 5: Linear Regression
print("\n5. Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)

results['Linear Regression'] = {'model': lr_model, 'predictions': lr_pred, 'RMSE': lr_rmse, 'MAE': lr_mae, 'R2': lr_r2, 'scaled': True}
print(f"   RMSE: {lr_rmse:,.2f} | R²: {lr_r2:.4f}")

# --- 7. MODEL COMPARISON ---
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

comparison_df = pd.DataFrame({
    model_name: {
        'RMSE': stats['RMSE'],
        'MAE': stats['MAE'],
        'R² Score': stats['R2']
    }
    for model_name, stats in results.items()
}).T

comparison_df = comparison_df.sort_values('R² Score', ascending=False)
print(f"\n{comparison_df.to_string()}")

# Find best model
best_model_name = comparison_df['R² Score'].idxmax()
best_model_data = results[best_model_name]
best_model = best_model_data['model']

print(f"\n🏆 BEST MODEL: {best_model_name}")

# --- 8. FEATURE IMPORTANCE ---
if best_model_name in ['Gradient Boosting', 'Random Forest']:
    print(f"\n{best_model_name} Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': predictors,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))

# --- 9. SAVE THE BEST MODEL ---
print(f"\n{'='*60}")
print("SAVING MODEL")
print(f"{'='*60}")

# Save the best model
joblib.dump(best_model, 'gb_price_predictor.pkl')
print(f"✅ Best model ({best_model_name}) saved as 'gb_price_predictor.pkl'")

# Save the scaler if needed
if best_model_data['scaled']:
    joblib.dump(scaler, 'scaler.pkl')
    print(f"✅ Scaler saved as 'scaler.pkl' (required for {best_model_name})")
else:
    joblib.dump(None, 'scaler.pkl')
    print(f"ℹ️  No scaler needed for {best_model_name}")

# Save metadata
metadata = {
    'model_name': best_model_name,
    'requires_scaling': best_model_data['scaled'],
    'r2_score': best_model_data['R2'],
    'rmse': best_model_data['RMSE'],
    'mae': best_model_data['MAE'],
    'predictors': predictors
}
joblib.dump(metadata, 'model_metadata.pkl')
print(f"✅ Model metadata saved as 'model_metadata.pkl'")