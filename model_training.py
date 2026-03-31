import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import numpy as np
import warnings
import os

# --- NEW IMPORTS FOR METRICS ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib
matplotlib.use('Agg') # Runs quietly in the background
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("--- LBS Hybrid Model Training Script Started ---")

if not os.path.exists('models'):
    os.makedirs('models')
    print("Created 'models' directory.")

# 1. Load your cleaned data
try:
    df = pd.read_csv('cleaned_data.csv')
    print("Successfully loaded cleaned_data.csv")
except FileNotFoundError:
    print("ERROR: 'cleaned_data.csv' not found.")
    exit()

# 2. Define Feature Sets
lifestyle_features = [
    'HoursOfSleep', 
    'AlcoholFrequency', 
    'VigorousActivity', 
    'ModerateActivity', 
    'SmokesNow',        
    'Energy_kcal', 
    'Sugar_gm', 
    'TotalFat_gm'
]

biometric_features = [
    'Age', 
    'BMI', 
    'FastingGlucose_mg_dL', 
    'TotalCholesterol_mg_dL',
    'SystolicBloodPressure', 
    'DiastolicBloodPressure', 
    'GlycoHemoglobin_A1c_percent',
    'Gender'            
]

target = 'DiagnosedHighBP' 

# 3. Preprocessing (CRITICAL)
print("Preprocessing data...")

df[target] = df[target].apply(lambda x: 1 if x == 'Yes' else 0)

activity_map = {'Not at all': 0, 'No': 0, 'Yes': 1, 'Several days': 2, 'Every day': 3}
gender_map = {'Male': 0, 'Female': 1}
smoke_map = {'Not at all': 0, 'No': 0, 'Yes': 1}

df['VigorousActivity'] = df['VigorousActivity'].map(activity_map)
df['ModerateActivity'] = df['ModerateActivity'].map(activity_map)
df['SmokesNow'] = df['SmokesNow'].map(smoke_map)
df['Gender'] = df['Gender'].map(gender_map)

all_features = lifestyle_features + biometric_features
for col in all_features:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    else:
        print(f"Warning: Column '{col}' not found in CSV. Skipping.")
        if col in lifestyle_features: lifestyle_features.remove(col)
        if col in biometric_features: biometric_features.remove(col)

df.dropna(subset=[target], inplace=True)
print(f"Using {len(df)} complete rows for training.")


# --- STAGE 1: Train K-Means Lifestyle Model ---
print("Training Stage 1 (K-Means Lifestyle Model)...")
scaler = StandardScaler()
df_lifestyle = df[lifestyle_features]
df_lifestyle_scaled = scaler.fit_transform(df_lifestyle)

kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_model.fit(df_lifestyle_scaled)

df['Lifestyle_Cluster'] = kmeans_model.labels_

joblib.dump(scaler, 'models/lifestyle_scaler.pkl')
joblib.dump(kmeans_model, 'models/kmeans_model.pkl')
print("Stage 1 Models saved.")


# --- STAGE 2: Train XGBoost Biometric Model ---
print("Training Stage 2 (XGBoost Biometric Model)...")
X_bio = df[biometric_features]
y = df[target]

X_bio_train, X_bio_test, y_train, y_test = train_test_split(X_bio, y, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_bio_train, y_train)

df['Biometric_Risk_Prob'] = xgb_model.predict_proba(df[biometric_features])[:, 1]

joblib.dump(xgb_model, 'models/xgb_model.pkl')
print("Stage 2 Model saved.")


# --- STAGE 3: Train Final "Meta-Model" ---
print("Training Stage 3 (Final Meta-Model)...")

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cluster_encoded = encoder.fit_transform(df[['Lifestyle_Cluster']])

cluster_cols = encoder.get_feature_names_out(['Lifestyle_Cluster'])

df_encoded = pd.DataFrame(cluster_encoded, columns=cluster_cols, index=df.index)

df_final_features = pd.concat([df['Biometric_Risk_Prob'], df_encoded], axis=1)
X_final = df_final_features
y = df[target]

X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

final_model = LogisticRegression(random_state=42)
final_model.fit(X_final_train, y_final_train)

joblib.dump(encoder, 'models/cluster_encoder.pkl')
joblib.dump(final_model, 'models/lbs_final_model.pkl')
print("Stage 3 Models saved.")
print("--- LBS Hybrid Model Training Complete. All 5 models saved to 'models/' folder. ---")

# =====================================================================
# --- EVALUATION: CONFUSION MATRIX & METRICS ---
# =====================================================================
print("\n" + "="*50)
print("--- Final Meta-Model Performance ---")

# 1. Make predictions
y_final_pred = final_model.predict(X_final_test)

# 2. Calculate Confusion Matrix
cm = confusion_matrix(y_final_test, y_final_pred)

print("\nRaw Confusion Matrix Results:")
print("                 Predicted Low Risk | Predicted High Risk")
print(f"Actual Low Risk  |        {cm[0][0]}        |         {cm[0][1]}")
print(f"Actual High Risk |        {cm[1][0]}        |         {cm[1][1]}")

# 3. Calculate Advanced Metrics
accuracy = accuracy_score(y_final_test, y_final_pred)
precision = precision_score(y_final_test, y_final_pred)
recall = recall_score(y_final_test, y_final_pred)
f1 = f1_score(y_final_test, y_final_pred)

print("\nDetailed Metrics:")
print(f"Accuracy:  {accuracy:.4f}  (Overall correctness)")
print(f"Precision: {precision:.4f}  (When it predicts High Risk, how often is it actually right?)")
print(f"Recall:    {recall:.4f}  (Out of ALL actual High Risk cases, how many did it successfully catch?)")
print(f"F1 Score:  {f1:.4f}  (The harmonic balance between Precision and Recall)")

# 4. Save Image
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Risk', 'High Risk'])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title('Confusion Matrix - Final Meta-Model')

plot_path = 'static/confusion_matrix_result.png'
plt.savefig(plot_path, bbox_inches='tight')
print(f"\n✅ Graphic saved successfully to: '{plot_path}'")
print("="*50 + "\n")
# 5. Generate and Save AUC-ROC Curve (Dark Theme for Webpage)
print("\n--- Generating AUC-ROC Curve ---")

y_final_prob = final_model.predict_proba(X_final_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_final_test, y_final_prob)
roc_auc = auc(fpr, tpr)

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

# ROC curve
ax.plot(
    fpr,
    tpr,
    color='#f97316',
    linewidth=3,
    label=f'ROC Curve (AUC = {roc_auc:.3f})'
)

# diagonal reference
ax.plot(
    [0, 1],
    [0, 1],
    color='#00b0ff',
    linewidth=2,
    linestyle='--',
    label='Random Guess'
)

# formatting
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title('Receiver Operating Characteristic (ROC Curve)', fontsize=16)

ax.tick_params(axis='both', labelsize=12)

ax.grid(True, linestyle='--', alpha=0.4)

ax.legend(loc='lower right', fontsize=12)

plt.tight_layout()

if not os.path.exists('static'):
    os.makedirs('static')

roc_path = 'static/roc_curve.png'
plt.savefig(roc_path, bbox_inches='tight', transparent=True)
plt.close()

print(f"✅ ROC Curve image saved successfully to: '{roc_path}'")