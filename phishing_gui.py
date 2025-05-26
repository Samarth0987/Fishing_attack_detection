import pandas as pd
import numpy as np
import streamlit as st
from urllib.parse import urlparse
import re
from xgboost import XGBClassifier
import joblib
import os

# === Step 1: Load Dataset ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('phishing_dataset.csv')
        df.columns = df.columns.str.strip().str.lower()
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        df['label'] = df['label'].fillna('legitimate')
        df['label'] = df['label'].map({'bad': 1, 'good': 0, 'legitimate': 0})
        df = df.dropna(subset=['label'])
        return df
    except FileNotFoundError:
        st.error("❌ File 'phishing_dataset.csv' not found.")
        return pd.DataFrame()

# === Step 2: Feature Engineering ===
def extract_features(url):
    parsed = urlparse(url)
    return {
        'url_length': len(url),
        'has_https': int(parsed.scheme == 'https'),
        'num_dots': url.count('.'),
        'has_at': int('@' in url),
        'has_dash': int('-' in parsed.netloc),
        'has_ip': int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', url))),
    }

# === Step 3: Load or Train Model ===
@st.cache_resource
def load_model(df):
    model_path = "phishing_model.pkl"

    features_df = df['url'].apply(extract_features).apply(pd.Series)
    features_df['label'] = df['label']
    X = features_df.drop('label', axis=1)
    y = features_df['label']

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = XGBClassifier(eval_metric='logloss', verbosity=1)
        model.fit(X, y)
        joblib.dump(model, model_path)
    return model

# === Streamlit UI ===
st.set_page_config(page_title="Phishing URL Detector", layout="centered")
st.title("🛡️ Phishing URL Detector")
st.markdown("Enter a URL below and we'll check if it's likely **legitimate** or **phishing**.")

url_input = st.text_input("🔗 Enter a URL:", "")

# Load data and model only when needed
df = load_data()
if not df.empty:
    model = load_model(df)

    if st.button("🔍 Check URL"):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                features = extract_features(url_input)
                input_df = pd.DataFrame([features])
                prediction = model.predict(input_df)[0]

                if prediction == 1:
                    st.error("🚨 Warning: This URL is likely **phishing**.")
                else:
                    st.success("✅ This URL is likely **legitimate**.")
            except Exception as e:
                st.exception(f"Error while predicting: {e}")

