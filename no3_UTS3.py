#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import joblib
import numpy as np

# Fungsi untuk memuat model dan encoder
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load('xgboost_model.pkl')
        label_encoder_gender = joblib.load('label_encoder_gender.pkl')
        label_encoder_defaults = joblib.load('label_encoder_defaults.pkl')
        label_encoder_home = joblib.load('label_encoder_home.pkl')
        label_encoder_loan_intent = joblib.load('label_encoder_loan_intent.pkl')
        ordinal_encoder_education = joblib.load('ordinal_encoder_education.pkl')
        return model, label_encoder_gender, label_encoder_defaults, label_encoder_home, label_encoder_loan_intent, ordinal_encoder_education
    except Exception as e:
        st.error(f"❌ Gagal memuat model/encoder: {e}")
        return None, None, None, None, None, None

def main():
    st.set_page_config(page_title="Loan Status Prediction", layout="centered")
    st.title("📊 Loan Booking Status Prediction")

    model, le_gender, le_defaults, le_home, le_intent, ord_edu = load_model_and_encoders()
    if model is None:
        return  # Stop if model failed to load

    # Input dari pengguna
    st.header("📝 Masukkan Data Peminjam")
    age = st.number_input("Person Age", 18, 100, 30)
    gender = st.selectbox("Person Gender", ['Male', 'Female'])
    education = st.selectbox("Person Education", ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
    income = st.number_input("Person Income", 0, 1000000, 50000)
    exp = st.number_input("Employment Experience (years)", 0, 50, 5)
    home = st.selectbox("Home Ownership", ['OWN', 'RENT', 'MORTGAGE', 'OTHER'])
    amount = st.number_input("Loan Amount", 1000, 35000, 10000)
    intent = st.selectbox("Loan Intent", ['EDUCATION', 'DEBTCONSOLIDATION', 'VENTURE', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT'])
    rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 12.0)
    percent_income = st.number_input("Loan % of Income", 0.0, 1.0, 0.2)
    history = st.number_input("Credit History Length", 0, 30, 5)
    credit = st.number_input("Credit Score", 400, 850, 650)
    defaults = st.selectbox("Previous Loan Defaults", ['Yes', 'No'])

    if st.button("🔍 Prediksi Status"):
        try:
            # Encoding input
            gender_enc = le_gender.transform([gender])[0]
            edu_enc = ord_edu.transform([[education]])[0][0]
            home_enc = le_home.transform([home])[0]
            intent_enc = le_intent.transform([intent])[0]
            default_enc = le_defaults.transform([defaults])[0]

            # Membuat array fitur
            features = np.array([
                age, gender_enc, edu_enc, income, exp,
                home_enc, amount, intent_enc, rate,
                percent_income, history, credit, default_enc
            ]).reshape(1, -1)

            # Prediksi
            pred = model.predict(features)
            result = "✅ Approved" if pred[0] == 1 else "❌ Rejected"
            st.success(f"Hasil Prediksi: {result}")

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")

if __name__ == '__main__':
    main()



# In[ ]:





# In[ ]:




