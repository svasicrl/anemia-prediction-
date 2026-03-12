!pip install streamlit
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load the trained model, scaler, and label encoder
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('le_gender.pkl')
model_features = joblib.load('model_features.pkl')
age_group_categories = joblib.load('age_group_categories.pkl')

st.title('Anemia Prediction Web Application')
st.write('Enter patient details to predict anemia type.')

# Input widgets for features
gender_options = ['Male', 'Female']
gender_input = st.radio('Gender', gender_options)
age_input = st.slider('Age', 0, 90, 30) # Assuming age range 0-90
age_group_input = st.selectbox('Age Group', age_group_categories)
hb_input = st.number_input('HB (g/dL)', min_value=0.0, max_value=20.0, value=12.0, step=0.1)
mcv_input = st.number_input('MCV (fL)', min_value=50.0, max_value=120.0, value=90.0, step=0.1)
rbc_input = st.number_input('RBC (million cells/µL)', min_value=2.0, max_value=7.0, value=4.5, step=0.01)
pcv_input = st.number_input('PCV (%)', min_value=20.0, max_value=60.0, value=40.0, step=0.1)
rdw_input = st.number_input('RDW (%)', min_value=10.0, max_value=25.0, value=14.0, step=0.1)
mch_input = st.number_input('MCH (pg/RBC)', min_value=10.0, max_value=40.0, value=28.0, step=0.1)
mchc_input = st.number_input('MCHC (g/dL)', min_value=20.0, max_value=40.0, value=33.0, step=0.1)
reticulocyte_count_input = st.number_input('Reticulocyte Count (%)', min_value=0.0, max_value=5.0, value=1.0, step=0.01)
green_king_index_input = st.number_input('Green & King Index', min_value=0.0, max_value=300.0, value=100.0, step=0.1)

if st.button('Predict Anemia Type'):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([{
        'Gender': gender_input,
        'Age': age_input,
        'Age_Group': age_group_input,
        'HB (g/dL)': hb_input,
        'MCV (fL)': mcv_input,
        'RBC (million cells/µL)': rbc_input,
        'PCV (%)': pcv_input,
        'RDW (%)': rdw_input,
        'MCH (pg/RBC)': mch_input,
        'MCHC (g/dL)': mchc_input,
        'Reticulocyte Count (%)': reticulocyte_count_input,
        'Green & King Index': green_king_index_input
    }])

    # Preprocess the input data
    # 1. Encode 'Gender'
    input_data['Gender'] = le_gender.transform(input_data['Gender'])

    # 2. One-hot encode 'Age_Group'
    # Create all possible Age_Group columns with default False, then set the correct one to True
    for category in age_group_categories:
        if category != 'Elderly': # drop_first=True was used, so 'Elderly' would be the reference category
            col_name = f'Age_Group_{category}'
            input_data[col_name] = (input_data['Age_Group'] == category)
    input_data = input_data.drop(columns=['Age_Group'])

    # 3. Scale numerical features
    numerical_cols = [col for col in model_features if col not in ['Gender', 'Age_Group_Adult', 'Age_Group_Child']]
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    # Ensure the order of columns matches the training data
    input_data = input_data[model_features]

    # Make prediction
    prediction = model.predict(input_data)

    st.success(f'Predicted Anemia Type: {prediction[0]}')
