import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")
df = pd.read_csv('datasets/FInal Dataset/dataset_for_training.csv')
print(f"Initial dataset shape: {df.shape}")
print(f"\nDataset columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
if 'month/year' in df.columns:
    df['year'] = pd.to_datetime(df['month/year'], format='%m/%Y').dt.year
    df = df.drop('month/year', axis=1)

if 'ZHVI' in df.columns:
    df = df.rename(columns={'ZHVI': 'avg_house_price'})

df.replace('N/A', np.nan, inplace=True)

numeric_columns = ['year', 'zip', 'median_income', 'crime_rate_per_1000', 'business_count', 'avg_house_price']
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

X = df[['year', 'zip', 'median_income', 'crime_rate_per_1000', 'business_count']]
y = df['avg_house_price']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
results = {}

# 1. Random Forest Regressor
print("\n" + "="*60)
print("Training Random Forest Regressor...")
print("="*60)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

results['Random Forest'] = {
    'MSE': rf_mse,
    'RMSE': rf_rmse,
    'MAE': rf_mae,
    'R2': rf_r2
}

print(f"Random Forest Results:")
print(f"  MSE: {rf_mse:,.2f}")
print(f"  RMSE: {rf_rmse:,.2f}")
print(f"  MAE: {rf_mae:,.2f}")
print(f"  R² Score: {rf_r2:.4f}")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nFeature Importance:")
print(feature_importance)

# 2. Lasso Regression
print("\n" + "="*60)
print("Training Lasso Regression...")
print("="*60)
lasso_model = Lasso(alpha=1.0, random_state=42, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)
lasso_pred = lasso_model.predict(X_test_scaled)

lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

results['Lasso'] = {
    'MSE': lasso_mse,
    'RMSE': lasso_rmse,
    'MAE': lasso_mae,
    'R2': lasso_r2
}

print(f"Lasso Results:")
print(f"  MSE: {lasso_mse:,.2f}")
print(f"  RMSE: {lasso_rmse:,.2f}")
print(f"  MAE: {lasso_mae:,.2f}")
print(f"  R² Score: {lasso_r2:.4f}")

# Lasso coefficients
lasso_coef = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lasso_model.coef_
}).sort_values('coefficient', ascending=False)
print(f"\nLasso Coefficients:")
print(lasso_coef)

# 3. Ridge Regression
print("\n" + "="*60)
print("Training Ridge Regression...")
print("="*60)
ridge_model = Ridge(alpha=1.0, random_state=42, max_iter=10000)
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)

ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

results['Ridge'] = {
    'MSE': ridge_mse,
    'RMSE': ridge_rmse,
    'MAE': ridge_mae,
    'R2': ridge_r2
}

print(f"Ridge Results:")
print(f"  MSE: {ridge_mse:,.2f}")
print(f"  RMSE: {ridge_rmse:,.2f}")
print(f"  MAE: {ridge_mae:,.2f}")
print(f"  R² Score: {ridge_r2:.4f}")

# Ridge coefficients
ridge_coef = pd.DataFrame({
    'feature': X.columns,
    'coefficient': ridge_model.coef_
}).sort_values('coefficient', ascending=False)
print(f"\nRidge Coefficients:")
print(ridge_coef)

# 4. Linear Regression
print("\n" + "="*60)
print("Training Linear Regression...")
print("="*60)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

lr_mse = mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

results['Linear Regression'] = {
    'MSE': lr_mse,
    'RMSE': lr_rmse,
    'MAE': lr_mae,
    'R2': lr_r2
}

print(f"Linear Regression Results:")
print(f"  MSE: {lr_mse:,.2f}")
print(f"  RMSE: {lr_rmse:,.2f}")
print(f"  MAE: {lr_mae:,.2f}")
print(f"  R² Score: {lr_r2:.4f}")

# Linear Regression coefficients
lr_coef = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_
}).sort_values('coefficient', ascending=False)
print(f"\nLinear Regression Coefficients:")
print(lr_coef)

# 5. Gradient Boosting Regressor
print("\n" + "="*60)
print("Training Gradient Boosting Regressor...")
print("="*60)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

gb_mse = mean_squared_error(y_test, gb_pred)
gb_rmse = np.sqrt(gb_mse)
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_r2 = r2_score(y_test, gb_pred)

results['Gradient Boosting'] = {
    'MSE': gb_mse,
    'RMSE': gb_rmse,
    'MAE': gb_mae,
    'R2': gb_r2
}

print(f"Gradient Boosting Results:")
print(f"  MSE: {gb_mse:,.2f}")
print(f"  RMSE: {gb_rmse:,.2f}")
print(f"  MAE: {gb_mae:,.2f}")
print(f"  R² Score: {gb_r2:.4f}")

# Feature importance for Gradient Boosting
gb_feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nGradient Boosting Feature Importance:")
print(gb_feature_importance)

# Summary comparison
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
comparison_df = pd.DataFrame(results).T
print(comparison_df)

# Find best model
best_model_r2 = comparison_df['R2'].idxmax()
best_model_rmse = comparison_df['RMSE'].idxmin()

print(f"\nBest model by R² Score: {best_model_r2} (R² = {comparison_df.loc[best_model_r2, 'R2']:.4f})")
print(f"Best model by RMSE: {best_model_rmse} (RMSE = {comparison_df.loc[best_model_rmse, 'RMSE']:,.2f})")

