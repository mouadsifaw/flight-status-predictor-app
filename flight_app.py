import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the preprocessor and model
try:
    with open('preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
    print("File loaded successfully.")
except Exception as e:
    print(f"Error loading file: {e}")

try:
    with open('best_rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("File loaded successfully.")
except Exception as e:
    print(f"Error loading file: {e}")
    
# CSS to inject contained in a string
page_bg_img = '''
<style>
body {
background-image: url("https://www.example.com/your-image.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Define the user interface
st.markdown("<h1 style='text-align: center; color: white;'>Flight Delay Prediction</h1>", unsafe_allow_html=True)

# Input fields
st.markdown("<h3 style='color: white;'>Select the flight details:</h3>", unsafe_allow_html=True)

year = st.selectbox('Year', [2023, 2024], key='year', index=0)
month = st.selectbox('Month', list(range(1, 13)), key='month', index=0)
day = st.selectbox('Day', list(range(1, 32)), key='day', index=0)
dep_time_block = st.selectbox('Departure Time Block', [
    'Night', 'Early Morning', 'Evening', 'Morning', 'Afternoon', 'Early Afternoon'], key='dep_time_block', index=0)
carrier = st.selectbox('Carrier', [
    'Southwest Airlines Co.', 'United Air Lines Inc.', 'American Airlines Inc.',
    'Spirit Air Lines', 'SkyWest Airlines Inc.', 'Delta Air Lines Inc.',
    'Endeavor Air Inc.', 'PSA Airlines Inc.', 'Envoy Air',
    'Hawaiian Airlines Inc.', 'Republic Airline', 'JetBlue Airways',
    'Allegiant Air', 'Frontier Airlines Inc.', 'Alaska Airlines Inc.'
], key='carrier', index=0)

# Predict button
if st.button('Predict', key='predict'):
    # Prepare the features as a DataFrame
    features = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Dep_Time_Block_Group': [dep_time_block],
        'Carrier': [carrier]
    })
    
    # Preprocess the features
    preprocessed_features = preprocessor.transform(features)
    
    # Make prediction
    prediction = model.predict(preprocessed_features)
    
    # Display the result
    if prediction[0] == 1:
        st.markdown("<h3 style='color: white;'>The flight will likely be delayed by 15 minutes or more.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: white;'>The flight will likely not be delayed by 15 minutes or more.</h3>", unsafe_allow_html=True)

