import streamlit as st
import pickle
import pandas as pd
import requests
from io import BytesIO

# Function to download file from Google Drive
def download_file_from_drive(file_id):
    base_url = 'https://drive.google.com/uc'
    url = f'{base_url}?id={file_id}'
    response = requests.get(url)
    response.raise_for_status()  # Check that the request was successful
    return BytesIO(response.content)

# Load the preprocessor
preprocessor = None
try:
    # Assuming preprocessor.pkl is in the same directory as this script
    with open('preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
    st.success("Preprocessor loaded successfully.")
except Exception as e:
    st.error(f"Error loading preprocessor: {e}")

# Google Drive file ID for the model
model_file_id = '1OzFP3cFKwKbGVMW35ytUv7CppVKfnEIs'

# Load the model
best_rf = None
try:
    model_file = download_file_from_drive(model_file_id)
    best_rf = pickle.load(model_file)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

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
    if preprocessor is None or best_rf is None:
        st.error("Model or preprocessor not loaded.")
    else:
        try:
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
            prediction = best_rf.predict(preprocessed_features)

            # Display the result
            if prediction[0] == 1:
                st.markdown("<h3 style='color: white;'>The flight will likely be delayed by 15 minutes or more.</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color: white;'>The flight will likely not be delayed by 15 minutes or more.</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
