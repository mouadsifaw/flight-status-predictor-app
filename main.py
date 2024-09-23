import streamlit as st
import pickle
import pandas as pd

# Load the preprocessor and model
try:
    with open('my_preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
    st.write("Preprocessor loaded successfully.")
except Exception as e:
    st.error(f"Error loading preprocessor: {e}")

try:
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define the user interface
st.markdown("<h1 style='text-align: center;'>Flight Delay Prediction</h1>", unsafe_allow_html=True)

# Input fields
year = st.selectbox('Year', [2024, 2025])
month = st.selectbox('Month', list(range(1, 13)))
day = st.selectbox('Day', list(range(1, 32)))
dep_time_block = st.selectbox('Departure Time Block', [
    'Night', 'Early Morning', 'Evening', 'Morning', 'Afternoon', 'Early Afternoon'])
carrier = st.selectbox('Carrier', [
    'Southwest Airlines Co.', 'United Air Lines Inc.', 'American Airlines Inc.',
    'Spirit Air Lines', 'SkyWest Airlines Inc.', 'Delta Air Lines Inc.',
    'Endeavor Air Inc.', 'PSA Airlines Inc.', 'Envoy Air',
    'Hawaiian Airlines Inc.', 'Republic Airline', 'JetBlue Airways',
    'Allegiant Air', 'Frontier Airlines Inc.', 'Alaska Airlines Inc.'])

# Predict button
if st.button('Predict'):
    # Prepare the features as a DataFrame
    features = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Dep_Time_Block_Group': [dep_time_block],
        'Carrier': [carrier]
    })
    
    # Preprocess the features
    try:
        preprocessed_features = preprocessor.transform(features)
    except Exception as e:
        st.error(f"Error preprocessing features: {e}")
    
    # Make prediction
    try:
        prediction = model.predict(preprocessed_features)
        if prediction[0] == 1:
            st.write("The flight will likely be delayed by 15 minutes or more.")
        else:
            st.write("The flight will likely not be delayed by 15 minutes or more.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

