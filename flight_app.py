import streamlit as st
import pickle
import pandas as pd
import calendar

# Load the preprocessor and model
try:
    with open('preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
except Exception as e:
    st.error(f"Error loading preprocessor: {e}")

try:
    with open('best_rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define the user interface
st.markdown("""
    <style>
    .header {
        text-align: center;
        color: #1f77b4;
        font-size: 2em;
        margin-bottom: 20px;
    }
    .description {
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 20px;
    }
    footer {
        text-align: center;
        margin-top: 20px;
        font-size: 0.8em;
        color: #888888;
    }
    </style>
    <div class="header">Flight Delay Prediction</div>
    <div class="description">Enter flight details to predict if it will be delayed by 15 minutes or more.</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header('Flight Delay Prediction')
st.sidebar.write("Use this app to predict whether a flight will be delayed by 15 minutes or more based on various input features.")

# Input fields in columns
col1, col2 = st.columns(2)

with col1:
    year = st.selectbox('Year', [2024, 2025], help="Select the year of the flight.")
    month = st.selectbox('Month', list(range(1, 13)), help="Select the month of the flight.")
