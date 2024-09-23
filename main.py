import streamlit as st
import pickle
import pandas as pd

# Load the preprocessor and model
try:
    with open('my_preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
    st.success("Preprocessor loaded successfully.")
except Exception as e:
    st.error(f"Error loading preprocessor: {e}")

try:
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# CSS styling for the app
st.markdown("""
    <style>
    body {
        background-color: #f0f4f8;
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        background: #007BFF;
        color: white;
        padding: 15px;
        text-align: center;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .container {
        margin: 20px;
    }
    .input-box {
        margin: 10px 0;
    }
    .result {
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        color: #fff;
        border: 2px solid;
        text-align: center;
    }
    .positive-result {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .negative-result {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #555;
    }
    .legend {
        padding: 15px;
        background: #e9ecef;
        border-radius: 10px;
        border: 1px solid #ced4da;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .legend-item {
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("<div class='header'><h1>Flight Delay Prediction</h1></div>", unsafe_allow_html=True)

# Container for input fields
st.markdown("<div class='container'>", unsafe_allow_html=True)

# Input fields
year = st.selectbox('Year', [2024, 2025], key='year', help='Select the year of the flight.')
month = st.selectbox('Month', list(range(1, 13)), key='month', help='Select the month of the flight.')
day = st.selectbox('Day', list(range(1, 32)), key='day', help='Select the day of the flight.')
dep_time_block = st.selectbox('Departure Time Block', [
    'Night', 'Early Morning', 'Evening', 'Morning', 'Afternoon', 'Early Afternoon'], key='dep_time_block', help='Select the departure time block.')
carrier = st.selectbox('Carrier', [
    'Southwest Airlines Co.', 'United Air Lines Inc.', 'American Airlines Inc.',
    'Spirit Air Lines', 'SkyWest Airlines Inc.', 'Delta Air Lines Inc.',
    'Endeavor Air Inc.', 'PSA Airlines Inc.', 'Envoy Air',
    'Hawaiian Airlines Inc.', 'Republic Airline', 'JetBlue Airways',
    'Allegiant Air', 'Frontier Airlines Inc.', 'Alaska Airlines Inc.'], key='carrier', help='Select the airline carrier.')

# Display the legend
st.markdown("""
    <div class="legend">
        <div class="legend-item"><strong>Night:</strong> 12:00 AM - 6:00 AM</div>
        <div class="legend-item"><strong>Early Morning:</strong> 6:00 AM - 9:00 AM</div>
        <div class="legend-item"><strong>Morning:</strong> 9:00 AM - 12:00 PM</div>
        <div class="legend-item"><strong>Early Afternoon:</strong> 12:00 PM - 3:00 PM</div>
        <div class="legend-item"><strong>Afternoon:</strong> 3:00 PM - 6:00 PM</div>
        <div class="legend-item"><strong>Evening:</strong> 6:00 PM - 12:00 AM</div>
    </div>
    """, unsafe_allow_html=True)

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
    try:
        preprocessed_features = preprocessor.transform(features)
    except Exception as e:
        st.error(f"Error preprocessing features: {e}")
    
    # Make prediction
    try:
        prediction = model.predict(preprocessed_features)
        if prediction[0] == 1:
            st.markdown("<p class='result negative-result'>The flight will likely be delayed by 15 minutes or more.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='result positive-result'>The flight will likely not be delayed by 15 minutes or more.</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error making prediction: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# Footer with developer name
st.markdown("<div class='footer'>Developed by MOUAD AIT KHOUYA</div>", unsafe_allow_html=True)
