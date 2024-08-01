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
    /* Responsive design */
    @media (max-width: 768px) {
        .header {
            font-size: 1.5em;
        }
        .description {
            font-size: 1em;
        }
        footer {
            font-size: 0.7em;
        }
    }
    </style>
    <div class="header">Flight Delay Prediction</div>
    <div class="description">Enter flight details to predict if it will be delayed by 15 minutes or more.</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header('Flight Delay Prediction')
st.sidebar.write("Use this app to predict whether a flight will be delayed by 15 minutes or more based on various input features.")

# Input fields in columns
col1, col2 = st.columns([1, 1])

with col1:
    year = st.selectbox(
        'Year',
        [2024, 2025],
        help="Select the year in which the flight is scheduled to take place. Only the years 2024 and 2025 are available."
    )
    month = st.selectbox(
        'Month',
        list(range(1, 13)),
        help="Select the month in which the flight is scheduled. Choose a number from 1 (January) to 12 (December)."
    )
    day = st.selectbox(
        'Day',
        list(range(1, 32)),
        help="Select the day of the month for the flight. Ensure the day is valid for the chosen month and year."
    )

with col2:
    dep_time_block = st.selectbox(
        'Departure Time Block',
        ['Night', 'Early Morning', 'Evening', 'Morning', 'Afternoon', 'Early Afternoon'],
        help="Select the time block when the flight is scheduled to depart. Options include Night, Early Morning, Morning, etc."
    )
    carrier = st.selectbox(
        'Carrier',
        [
            'Southwest Airlines Co.', 'United Air Lines Inc.', 'American Airlines Inc.',
            'Spirit Air Lines', 'SkyWest Airlines Inc.', 'Delta Air Lines Inc.',
            'Endeavor Air Inc.', 'PSA Airlines Inc.', 'Envoy Air',
            'Hawaiian Airlines Inc.', 'Republic Airline', 'JetBlue Airways',
            'Allegiant Air', 'Frontier Airlines Inc.', 'Alaska Airlines Inc.'
        ],
        help="Select the airline carrier for the flight. Choose from a list of major airlines."
    )

# Predict button
if st.button('Predict'):
    # Validate day for the chosen month and year
    if not (1 <= day <= calendar.monthrange(year, month)[1]):
        st.error(f"Day {day} is not valid for {calendar.month_name[month]} {year}.")
    else:
        with st.spinner('Making prediction...'):
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
                preprocessed_features = None
            
            if preprocessed_features is not None:
                # Make prediction
                try:
                    prediction = model.predict(preprocessed_features)
                    if prediction[0] == 1:
                        st.write("The flight will likely be delayed by 15 minutes or more.")
                    else:
                        st.write("The flight will likely not be delayed by 15 minutes or more.")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

# Footer
st.markdown("""
    <footer>
        Developed by Mouad Sifaw.
    </footer>
""", unsafe_allow_html=True)
