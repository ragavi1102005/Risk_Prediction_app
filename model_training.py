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
# --- These are the features your new dashboard will ask for ---
lifestyle_features = [
    'HoursOfSleep', 
    'AlcoholFrequency', # Assumed numeric 0-3 (Not at all, Once, Several, Every day)
    'VigorousActivity', # Assumed text: 'Yes', 'No', 'Not at all', etc.
    'ModerateActivity', # Assumed text
    'SmokesNow',        # Assumed text: 'Yes', 'No'
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
    'Gender'            # Assumed text: 'Male', 'Female'
]

# --- This is your prediction target ---
target = 'DiagnosedHighBP' # We'll predict High Blood Pressure

# 3. Preprocessing (CRITICAL)
print("Preprocessing data...")

# Convert target to 0s and 1s
df[target] = df[target].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert text features to consistent, model-friendly formats
activity_map = {'Not at all': 0, 'No': 0, 'Yes': 1, 'Several days': 2, 'Every day': 3}
gender_map = {'Male': 0, 'Female': 1}
smoke_map = {'Not at all': 0, 'No': 0, 'Yes': 1}

df['VigorousActivity'] = df['VigorousActivity'].map(activity_map)
df['ModerateActivity'] = df['ModerateActivity'].map(activity_map)
df['SmokesNow'] = df['SmokesNow'].map(smoke_map)
df['Gender'] = df['Gender'].map(gender_map)

# Handle missing data (impute with median for robustness)
all_features = lifestyle_features + biometric_features
for col in all_features:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    else:
        print(f"Warning: Column '{col}' not found in CSV. Skipping.")
        if col in lifestyle_features: lifestyle_features.remove(col)
        if col in biometric_features: biometric_features.remove(col)

# Drop any remaining rows where target is missing
df.dropna(subset=[target], inplace=True)
print(f"Using {len(df)} complete rows for training.")


# --- STAGE 1: Train K-Means Lifestyle Model ---
print("Training Stage 1 (K-Means Lifestyle Model)...")
# Scale lifestyle data for K-Means
scaler = StandardScaler()
df_lifestyle = df[lifestyle_features]
df_lifestyle_scaled = scaler.fit_transform(df_lifestyle)

# Train K-Means to find 4 "Lifestyle Archetypes"
kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_model.fit(df_lifestyle_scaled)

# Get the cluster label for every person
df['Lifestyle_Cluster'] = kmeans_model.labels_

# Save Stage 1 models
joblib.dump(scaler, 'models/lifestyle_scaler.pkl')
joblib.dump(kmeans_model, 'models/kmeans_model.pkl')
print("Stage 1 Models saved.")


# --- STAGE 2: Train XGBoost Biometric Model ---
print("Training Stage 2 (XGBoost Biometric Model)...")
X_bio = df[biometric_features]
y = df[target]

# Split data for Stage 2
X_bio_train, X_bio_test, y_train, y_test = train_test_split(X_bio, y, test_size=0.2, random_state=42)

# Train a powerful XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_bio_train, y_train)

# Get the "Biometric Risk Score" (probability) for every person
df['Biometric_Risk_Prob'] = xgb_model.predict_proba(df[biometric_features])[:, 1]

# Save Stage 2 model
joblib.dump(xgb_model, 'models/xgb_model.pkl')
print("Stage 2 Model saved.")


# --- STAGE 3: Train Final "Meta-Model" ---
print("Training Stage 3 (Final Meta-Model)...")

# The inputs for our final model are the OUTPUTS of Stage 1 and 2
# We must one-hot encode the cluster labels
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cluster_encoded = encoder.fit_transform(df[['Lifestyle_Cluster']])

# --- THIS IS THE FIX ---
# Use the encoder's built-in function to get the *exact* feature names
# This guarantees it will match in app.py
cluster_cols = encoder.get_feature_names_out(['Lifestyle_Cluster'])
# --- END OF FIX ---

df_encoded = pd.DataFrame(cluster_encoded, columns=cluster_cols, index=df.index)

# Combine the biometric risk score and cluster labels
df_final_features = pd.concat([df['Biometric_Risk_Prob'], df_encoded], axis=1)
X_final = df_final_features
y = df[target]

# Split and train the final, simple Logistic Regression model
X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

final_model = LogisticRegression(random_state=42)
final_model.fit(X_final_train, y_final_train)

# Save Stage 3 models
joblib.dump(encoder, 'models/cluster_encoder.pkl')
joblib.dump(final_model, 'models/lbs_final_model.pkl')
print("Stage 3 Models saved.")
print("--- LBS Hybrid Model Training Complete. All 5 models saved to 'models/' folder. ---")

