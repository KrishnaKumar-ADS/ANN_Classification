import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# --- Load the trained model and preprocessing objects ---
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- Streamlit App UI ---
st.title("Customer Churn Prediction 📊")

# --- User Input ---
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
credit_score = st.number_input('Credit Score', value=600)
balance = st.number_input('Balance', value=50000.0)
estimated_salary = st.number_input('Estimated Salary', value=100000.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])

if st.button('Predict Churn'):
    # --- Data Preparation ---
    
    # 1. Create a dictionary with the initial features
    input_features = {
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_credit_card == 'Yes' else 0],
        'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
        'EstimatedSalary': [estimated_salary],
        'Gender': [label_encoder_gender.transform([gender])[0]],
    }
    input_data = pd.DataFrame(input_features)

    # 2. One-hot encode the 'Geography' feature
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_cols)

    # 3. Combine the dataframes
    input_data = pd.concat([input_data, geo_encoded_df], axis=1)

    # --- FIX: Reorder columns to match the scaler's expectations ---
    # Get the feature names the scaler was trained on
    expected_columns = scaler.feature_names_in_
    
    # Reorder the input_data columns to match the training order
    input_data = input_data[expected_columns]
    
    # 4. Scale the data (this will now work correctly)
    input_data_scaled = scaler.transform(input_data)

    # 5. Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    
    st.write(f"**Churn Probability:** `{prediction_proba:.4f}`")

    if prediction_proba > 0.5:
        st.error('**Conclusion: The customer is likely to CHURN.**')
    else:
        st.success('**Conclusion: The customer is likely to STAY.**')