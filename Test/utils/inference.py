import joblib
import numpy as np
import pandas as pd


def load_models():
    clf1 = joblib.load("models/queen_presence_model.pkl")
    clf2 = joblib.load("models/queen_acceptance_model.pkl")
    iso_forest = joblib.load("models/bee_sound_anomaly_model.pkl")
    return clf1, clf2, iso_forest


def run_inference(clf1, clf2, iso_forest, w_temp, w_hum, h_temp, h_hum, mfccs):
    # === Combined Feature Vector for Model 1 & 2 ===
    all_features = np.concatenate(([h_temp, h_hum, w_temp, w_hum], mfccs))
    all_feature_names = ["hive temp", "hive humidity", "weather temp", "weather humidity"] + [f"mfcc_{i+1}" for i in range(len(mfccs))]
    features_df_all = pd.DataFrame([all_features], columns=all_feature_names)

    # === MFCC-only Feature Vector for Anomaly Model ===
    features_df_mfcc_only = pd.DataFrame([mfccs], columns=[f"mfcc_{i+1}" for i in range(len(mfccs))])

    queen_presence = clf1.predict(features_df_all)[0]
    queen_acceptance = clf2.predict(features_df_all)[0]
    anomaly = iso_forest.predict(features_df_mfcc_only)[0]  # -1 is anomaly

    return {
        "queen_presence": 'Yes' if queen_presence == 1 else 'No',
        "queen_acceptance": queen_acceptance,
        "anomaly": "Anomaly" if anomaly == -1 else "Normal",
    }
