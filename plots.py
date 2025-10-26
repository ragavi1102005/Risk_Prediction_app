import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("--- LBS Hybrid Model Plot Generation Script Started ---")

if not os.path.exists('static'):
    os.makedirs('static')
    print("Created 'static' directory for images.")

# --- Load Models and Encoders ---
print("Loading saved LBS models...")
try:
    xgb_model = joblib.load('xgb_model.pkl')
    final_model = joblib.load('lbs_final_model.pkl')
    encoder = joblib.load('cluster_encoder.pkl')
except FileNotFoundError:
    print("ERROR: Model files not found. Run 'model_training.py' first.")
    exit()

# --- PLOT 1: BIOMETRIC (XGBoost) FEATURE IMPORTANCE ---
print("Generating Plot 1: Biometric Risk Drivers (XGBoost)...")
biometric_features = [
    'Age', 'BMI', 'FastingGlucose_mg_dL', 'TotalCholesterol_mg_dL',
    'SystolicBloodPressure', 'DiastolicBloodPressure', 'GlycoHemoglobin_A1c_percent', 'Gender'
]
importances = xgb_model.feature_importances_
feature_importance_series = pd.Series(importances, index=biometric_features).sort_values(ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x=feature_importance_series, y=feature_importance_series.index, palette='viridis')
plt.title('Plot 1: Biometric Risk Drivers (Stage 2 - XGBoost Model)')
plt.xlabel('Importance Score')
plt.ylabel('Biometric Features')
xgb_plot_path = 'static/xgb_feature_importance.png'
plt.savefig(xgb_plot_path, bbox_inches='tight')
plt.close()
print(f"Biometric importance plot saved to {xgb_plot_path}")

# --- PLOT 2: HYBRID MODEL (Logistic Regression) COEFFICIENTS ---
print("Generating Plot 2: Final LBS Model Coefficients (The 'Proof')...")

# Get feature names from the final model
cluster_cols = [f'Cluster_{int(i)}' for i in encoder.categories_[0]]
final_feature_names = ['Biometric_Risk_Prob'] + cluster_cols

# Get the coefficients (the "importance") from the final model
coefficients = final_model.coef_[0]
coef_series = pd.Series(coefficients, index=final_feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=coef_series, y=coef_series.index, palette='coolwarm')
plt.title('Plot 2: Final LBS Hybrid Model Coefficients (Stage 3)')
plt.xlabel('Coefficient (Impact on Final Risk)')
plt.ylabel('Meta-Features')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
lbs_plot_path = 'static/lbs_model_coefficients.png'
plt.savefig(lbs_plot_path, bbox_inches='tight')
plt.close()
print(f"LBS coefficients plot saved to {lbs_plot_path}")

print("--- All plots generated successfully! ---")
print(f"Add '{xgb_plot_path}' and '{lbs_plot_path}' to your report.")
