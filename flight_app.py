import streamlit as st
import pickle
import pandas as pd

# Load the preprocessor and model
try:
    with open('preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
except Exception:
    st.error("There was a problem loading the preprocessor.")

try:
    with open('best_rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception:
    st.error("There was a problem loading the model.")

# Define the user interface
st.markdown("<h1 style='text-align: center;'>Flight Delay Prediction</h1>", unsafe_allow_html=True)

# Add Instructions in the Sidebar
st.sidebar.markdown("""
### Instructions
1. **Select the flight details**: Use the dropdown menus to input the flight year, month, day, departure time block, and carrier.
2. **Click on 'Predict'**: After filling in the details, click the **Predict** button to see if the flight is likely to be delayed.
3. **View Results**: The result will be displayed below the button.
""")

# Input fields
year = st.selectbox('Year', [2023, 2024])
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
    except Exception:
        st.error("There was a problem processing your input.")
        st.stop()  # Stop further execution if preprocessing fails
    
    # Make prediction
    try:
        prediction = model.predict(preprocessed_features)
        if prediction[0] == 1:
            st.success("The flight will likely be delayed by 15 minutes or more.")
        else:
            st.success("The flight will likely not be delayed by 15 minutes or more.")
    except Exception:
        st.error("There was a problem making the prediction.")
