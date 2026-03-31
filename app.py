from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import warnings
import os
import shap

# --- NEW IMPORTS FOR SHAP GRAPH ---
import matplotlib
matplotlib.use('Agg') # CRITICAL: Tells matplotlib to run in the background
import matplotlib.pyplot as plt
import io
import base64

# Suppress warnings for a cleaner terminal
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enables frontend-backend communication

# --- Robust Helper Functions ---
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

# --- Clinical Override to Fix Age Bias ---
def apply_clinical_override(raw_prob, user_data):
    """
    Reduces the model's age bias by lowering the risk probability if 
    an older user has perfectly healthy clinical biometrics.
    """
    age = safe_float(user_data.get('Age', 0))
    sys_bp = safe_float(user_data.get('SystolicBloodPressure', 0))
    bmi = safe_float(user_data.get('BMI', 0))
    chol = safe_float(user_data.get('TotalCholesterol_mg_dL', 0))
    glucose = safe_float(user_data.get('FastingGlucose_mg_dL', 0))

    adjusted_prob = raw_prob

    if age >= 45 and raw_prob > 50:
        healthy_metrics = 0
        if 0 < sys_bp <= 125: healthy_metrics += 1
        if 0 < bmi <= 25: healthy_metrics += 1
        if 0 < chol <= 200: healthy_metrics += 1
        if 0 < glucose <= 100: healthy_metrics += 1

        if healthy_metrics == 4:
            adjusted_prob = raw_prob * 0.35  
        elif healthy_metrics == 3:
            adjusted_prob = raw_prob * 0.55  
        elif healthy_metrics == 2:
            adjusted_prob = raw_prob * 0.75  

    return round(adjusted_prob, 1)


# --- 1. Load All 5 Hybrid Model Files ---
try:
    scaler = joblib.load('models/lifestyle_scaler.pkl')
    kmeans_model = joblib.load('models/kmeans_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    encoder = joblib.load('models/cluster_encoder.pkl')
    final_model = joblib.load('models/lbs_final_model.pkl')
    print("--- All 5 hybrid models loaded successfully. ---")
except Exception as e:
    print(f"!!! CRITICAL ERROR: Failed to load models. {e}")
    print(f"!!! Error type: {type(e)}")
    print("!!! This is likely a Git LFS issue or corrupted .pkl file. !!!")
    exit()


# --- 2. Define Helper Mappings ---
activity_map = {'Not at all': 0, 'No': 0, 'Yes': 1, 'Several days': 2, 'Every day': 3}
gender_map = {'Male': 0, 'Female': 1}
smoke_map = {'Not at all': 0, 'No': 0, 'Yes': 1}
diet_map = {
    'balanced': {'Energy_kcal': 1800, 'Sugar_gm': 70, 'TotalFat_gm': 65},
    'low_fat': {'Energy_kcal': 1600, 'Sugar_gm': 80, 'TotalFat_gm': 40},
    'low_sugar': {'Energy_kcal': 1700, 'Sugar_gm': 30, 'TotalFat_gm': 70},
    'high_protein': {'Energy_kcal': 2000, 'Sugar_gm': 70, 'TotalFat_gm': 75}
}


# --- EXPLAINABILITY ENGINE (SHAP XAI LOGIC WITH GRAPH) ---
def generate_shap_explanation(model, input_df):
    """
    Generates a dynamic text explanation AND a base64-encoded SHAP bar graph.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]

        feature_names = input_df.columns
        contributions = list(zip(feature_names, vals))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        # 1. Generate Text
        top_risk = [f"{feat.replace('_', ' ')} (Impact: {val:.2f})" for feat, val in contributions if val > 0][:3]
        top_protect = [f"{feat.replace('_', ' ')} (Impact: {val:.2f})" for feat, val in contributions if val < 0][:3]

        if top_risk:
            text_exp = f"SHAP Analysis: Your risk score was primarily driven higher by: {', '.join(top_risk)}."
        else:
            text_exp = f"SHAP Analysis: Your risk score is well-managed, protected mostly by: {', '.join(top_protect)}."

        # 2. Generate SHAP Graph (Styled for Dark Mode)
        plt.style.use('dark_background')
        plt.figure(figsize=(8, 4))
        
        # Sort lowest to highest for the bar chart
        plot_data = sorted(contributions, key=lambda x: x[1])
        feats = [x[0].replace('_', ' ') for x in plot_data]
        impacts = [x[1] for x in plot_data]
        colors = ['#ef5350' if val > 0 else '#66bb6a' for val in impacts] # Red for Risk, Green for Healthy

        plt.barh(feats, impacts, color=colors)
        plt.xlabel("SHAP Value (Impact on Risk Score)")
        plt.title("Personalized SHAP Feature Impact")
        plt.axvline(0, color='grey', linewidth=0.8, linestyle='--')
        plt.tight_layout()

        # 3. Convert Graph to Base64 Image String
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        plt.close()
        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')

        return text_exp, base64_img

    except Exception as e:
        return f"SHAP explanation could not be generated due to an error: {str(e)}", None


# --- 3. Define Feature Order ---
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


# --- 4. Define Routes ---
@app.route('/')
def home():
    """Serves the main index.html page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receives data, runs the 3-stage hybrid model, and returns a prediction."""
    if request.is_json:
        json_data = request.get_json()
    else:
        json_data = request.form

    print(f"--- [DEBUG] Received data: {json_data} ---")

    try:
        # --- STAGE 1: Lifestyle Model ---
        diet_type = json_data.get('DietType', 'balanced')
        diet_values = diet_map.get(diet_type, diet_map['balanced'])

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
        df_lifestyle = df_lifestyle[lifestyle_features_order]
        scaled_lifestyle_data = scaler.transform(df_lifestyle)
        cluster_label = kmeans_model.predict(scaled_lifestyle_data)[0]

        # --- STAGE 2: Biometric Model ---
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
        df_biometric = df_biometric[biometric_features_order]
        biometric_risk_prob = xgb_model.predict_proba(df_biometric)[0, 1]

        # --- STAGE 3: Final Meta-Model ---
        cluster_label_2d = np.array([[cluster_label]])
        cluster_encoded = encoder.transform(cluster_label_2d)
        feature_names = encoder.get_feature_names_out(['Lifestyle_Cluster'])
        final_features_data = pd.DataFrame(cluster_encoded, columns=feature_names)
        final_features_data['Biometric_Risk_Prob'] = biometric_risk_prob

        final_model_cols = final_model.feature_names_in_
        final_features_data = final_features_data[final_model_cols]

        raw_probability_pct = final_model.predict_proba(final_features_data)[0, 1] * 100
        final_probability_pct = apply_clinical_override(raw_probability_pct, json_data)
        risk_category = 'Yes' if final_probability_pct >= 50 else 'No'

        print(f"[DEBUG] Raw Prob = {raw_probability_pct:.1f}%, Corrected Prob = {final_probability_pct}%")

        # --- SHAP XAI: Generate Dynamic Explanation AND Image ---
        explanation_text, shap_image_base64 = generate_shap_explanation(xgb_model, df_biometric)

        response = {
            'prediction': risk_category,
            'probability': final_probability_pct,
            'cluster': int(cluster_label),
            'explanation': explanation_text,
            'shap_image': shap_image_base64  
        }
        return jsonify(response)

    except Exception as e:
        print(f"!!! [ERROR] Prediction failed: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400


# --- 5. Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"--- Starting Flask server on port {port} ---")
    app.run(host='0.0.0.0', port=port, debug=False)