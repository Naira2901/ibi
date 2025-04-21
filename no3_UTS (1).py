#!/usr/bin/env python
# coding: utf-8

# In[2]:



# In[5]:


import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load('xgboost_model.pkl')
label_encoder_gender = joblib.load('label_encoder_gender.pkl')
label_encoder_defaults = joblib.load('label_encoder_defaults.pkl')
label_encoder_home = joblib.load('label_encoder_home.pkl')
label_encoder_loan_intent = joblib.load('label_encoder_loan_intent.pkl')
ordinal_encoder_education = joblib.load('ordinal_encoder_education.pkl')

def main():
    st.title('Loan Booking Status Prediction')

    # User input
    person_age = st.number_input('Person Age', 0, 100, 30)
    person_gender = st.selectbox('Person Gender', ['Male', 'Female'])
    person_education = st.selectbox('Person Education', ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
    person_income = st.number_input('Person Income', 0, 1000000, 50000)
    person_emp_exp = st.number_input('Person Employment Experience (years)', 0, 100, 5)
    person_home_ownership = st.selectbox('Person Home Ownership', ['OWN', 'RENT', 'MORTGAGE', 'OTHER'])
    loan_amnt = st.number_input('Loan Amount', 1000, 35000, 10000)
    loan_intent = st.selectbox('Loan Intent', ['EDUCATION', 'DEBTCONSOLIDATION', 'VENTURE', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT'])
    loan_int_rate = st.number_input('Loan Interest Rate', 0.0, 30.0, 12.0)
    loan_percent_income = st.number_input('Loan Percentage of Income', 0.0, 1.0, 0.2)
    cb_person_cred_hist_length = st.number_input('Credit History Length (years)', 0, 30, 5)
    credit_score = st.number_input('Credit Score', 400, 850, 650)
    previous_loan_defaults_on_file = st.selectbox('Previous Loan Defaults on File', ['Yes', 'No'])

    if st.button('Predict'):
        # Encode input
        gender_encoded = label_encoder_gender.transform([person_gender])[0]
        education_encoded = ordinal_encoder_education.transform([[person_education]])[0][0]
        home_encoded = label_encoder_home.transform([person_home_ownership])[0]
        intent_encoded = label_encoder_loan_intent.transform([loan_intent])[0]
        defaults_encoded = label_encoder_defaults.transform([previous_loan_defaults_on_file])[0]

        # Prepare features
        features = np.array([
            person_age,
            gender_encoded,
            education_encoded,
            person_income,
            person_emp_exp,
            home_encoded,
            loan_amnt,
            intent_encoded,
            loan_int_rate,
            loan_percent_income,
            cb_person_cred_hist_length,
            credit_score,
            defaults_encoded
        ]).reshape(1, -1)

        prediction = model.predict(features)
        st.success(f"Predicted Booking Status: {'Approved' if prediction[0] == 1 else 'Rejected'}")

if __name__ == '__main__':
    main()


# In[ ]:




