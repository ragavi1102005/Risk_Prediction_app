from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for a cleaner terminal
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # This enables Cross-Origin Resource Sharing

# --- NEW: Robust Helper Functions ---
def safe_float(value, default=0.0):
    """Safely converts a value to float, handling None and empty strings."""
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely converts a value to int, handling None and empty strings."""
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == '':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default
# --- End of Helper Functions ---


# --- 1. Load All 5 Hybrid Model Files ---
# All models are now loaded from the 'models/' subfolder
try:
    scaler = joblib.load('models/lifestyle_scaler.pkl')
    kmeans_model = joblib.load('models/kmeans_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    encoder = joblib.load('models/cluster_encoder.pkl')
    final_model = joblib.load('models/lbs_final_model.pkl')
    print("--- All 5 hybrid models loaded successfully. ---")
except FileNotFoundError as e:
    print(f"!!! CRITICAL ERROR: Model file not found. {e}")
    print("!!! Have you run 'python model_training.py' first? !!!")
    exit()

# --- 2. Define Helper Mappings (MUST match model_training.py) ---
activity_map = {'Not at all': 0, 'No': 0, 'Yes': 1, 'Several days': 2, 'Every day': 3}
gender_map = {'Male': 0, 'Female': 1}
smoke_map = {'Not at all': 0, 'No': 0, 'Yes': 1}
# New map for user-friendly diet input
diet_map = {
    'balanced': {'Energy_kcal': 1800, 'Sugar_gm': 70, 'TotalFat_gm': 65},
    'low_fat': {'Energy_kcal': 1600, 'Sugar_gm': 80, 'TotalFat_gm': 40},
    'low_sugar': {'Energy_kcal': 1700, 'Sugar_gm': 30, 'TotalFat_gm': 70},
    'high_protein': {'Energy_kcal': 2000, 'Sugar_gm': 70, 'TotalFat_gm': 75}
}

# --- 3. Define Feature Order (MUST match model_training.py) ---
lifestyle_features_order = [
    'HoursOfSleep', 'AlcoholFrequency', 'VigorousActivity', 
    'ModerateActivity', 'SmokesNow', 'Energy_kcal', 
    'Sugar_gm', 'TotalFat_gm'
]
biometric_features_order = [
    'Age', 'BMI', 'FastingGlucose_mg_dL', 'TotalCholesterol_mg_dL',
    'SystolicBloodPressure', 'DiastolicBloodPressure', 
    'GlycoHemoglobin_A1c_percent', 'Gender'
]

# --- 4. Define App Routes ---
@app.route('/')
def home():
    """Serves the main index.html page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives data, runs the 3-stage hybrid model, and returns a prediction."""
    
    # FIX for 400 ERROR: Check if data is JSON, otherwise use form
    if request.is_json:
        json_data = request.get_json()
    else:
        json_data = request.form
    
    print(f"--- [DEBUG] Received data: {json_data} ---")
    
    try:
        # --- STAGE 1: Run K-Means Lifestyle Model ---
        
        # 1a. Get user-friendly diet and map it to technical features
        diet_type = json_data.get('DietType', 'balanced')
        diet_values = diet_map.get(diet_type, diet_map['balanced'])
        
        # 1b. Create lifestyle DataFrame using the new safe functions
        lifestyle_data = {
            'HoursOfSleep': [safe_float(json_data.get('HoursOfSleep'), 7.0)],
            'AlcoholFrequency': [safe_int(json_data.get('AlcoholFrequency'), 1)],
            'VigorousActivity': [activity_map.get(json_data.get('VigorousActivity'), 0)],
            'ModerateActivity': [activity_map.get(json_data.get('ModerateActivity'), 0)],
            'SmokesNow': [smoke_map.get(json_data.get('SmokesNow'), 0)],
            'Energy_kcal': [diet_values['Energy_kcal']],
            'Sugar_gm': [diet_values['Sugar_gm']],
            'TotalFat_gm': [diet_values['TotalFat_gm']]
        }
        df_lifestyle = pd.DataFrame(lifestyle_data)
        
        # 1c. Enforce column order (KeyError fix) and scale
        df_lifestyle = df_lifestyle[lifestyle_features_order]
        scaled_lifestyle_data = scaler.transform(df_lifestyle)
        
        # 1d. Predict cluster
        cluster_label = kmeans_model.predict(scaled_lifestyle_data)[0]
        print(f"[DEBUG] Stage 1 Result: Identified Lifestyle Cluster {cluster_label}")

        # --- STAGE 2: Run XGBoost Biometric Model ---
        
        # 2a. Create biometric DataFrame using the new safe functions
        biometric_data = {
            'Age': [safe_int(json_data.get('Age'), 51)],
            'BMI': [safe_float(json_data.get('BMI'), 28.6)],
            'FastingGlucose_mg_dL': [safe_float(json_data.get('FastingGlucose_mg_dL'), 103.0)],
            'TotalCholesterol_mg_dL': [safe_float(json_data.get('TotalCholesterol_mg_dL'), 185.0)],
            'SystolicBloodPressure': [safe_float(json_data.get('SystolicBloodPressure'), 124.0)],
            'DiastolicBloodPressure': [safe_float(json_data.get('DiastolicBloodPressure'), 70.0)],
            'GlycoHemoglobin_A1c_percent': [safe_float(json_data.get('GlycoHemoglobin_A1c_percent'), 5.7)],
            'Gender': [gender_map.get(json_data.get('Gender'), 0)]
        }
        df_biometric = pd.DataFrame(biometric_data)
        
        # 2b. Enforce column order (KeyError fix) and predict risk probability
        df_biometric = df_biometric[biometric_features_order]
        biometric_risk_prob = xgb_model.predict_proba(df_biometric)[0, 1]
        print(f"[DEBUG] Stage 2 Result: Biometric Risk Probability {biometric_risk_prob:.4f}")

        # --- STAGE 3: Run LBS Meta-Model ---
        
        # 3a. One-hot encode the predicted cluster
        cluster_label_2d = np.array([[cluster_label]])
        cluster_encoded = encoder.transform(cluster_label_2d)
        
        # 3b. Create final feature DataFrame for the meta-model
        # Get column names *from the encoder* (matches training)
        feature_names = encoder.get_feature_names_out(['Lifestyle_Cluster'])
        final_features_data = pd.DataFrame(cluster_encoded, columns=feature_names)
        final_features_data['Biometric_Risk_Prob'] = biometric_risk_prob
        
        # --- THIS IS THE FIX ---
        # Get the *exact* feature order the model was trained on
        final_model_cols = final_model.feature_names_in_
        # Re-order our DataFrame to match that exact order
        final_features_data = final_features_data[final_model_cols]
        # --- END OF FIX ---
        
        # 3c. Get final prediction and probability
        final_prediction = final_model.predict(final_features_data)[0]
        final_probability_pct = round(final_model.predict_proba(final_features_data)[0, 1] * 100, 1)
        risk_category = 'Yes' if final_prediction == 1 else 'No'
        
        print(f"[DEBUG] Stage 3 Result: Final Risk = {risk_category}, Probability = {final_probability_pct}%")

        # --- 4. Send Response ---
        response = {
            'prediction': risk_category,
            'probability': final_probability_pct,
            'cluster': int(cluster_label)  # Send the cluster label back to the UI
        }
        return jsonify(response)

    except Exception as e:
        # This is the robust error handling that fixes the 400 error
        print(f"!!! [ERROR] Prediction failed: {str(e)}")
        print(f"Data that caused error: {json_data}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

# --- 5. Run the App ---
if __name__ == '__main__':
    print("--- Starting Flask server at http://127.0.0.1:5000/ ---")
    app.run(debug=True, port=5000)

