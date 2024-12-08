**Flight Delay Predictor App**

The Flight Delay Predictor App is a machine learning-powered web application designed to predict whether a flight will be delayed by at least 15 minutes. The app enables users to input flight details such as the carrier, date, and departure time block, and provides a real-time prediction of potential delays. 

***Key Features***

   - User-Friendly Interface:
        A clean and simple web-based interface for entering flight details.
        Dropdown menus for selecting the carrier and departure time block.

   - Real-Time Predictions:
        Instant predictions based on a pre-trained machine learning model.
        Indicates whether a flight is likely to be delayed by at least 15 minutes.

   - Data-Driven Insights:
        Powered by a Random Forest model trained on a dataset of 2.5 million flight records.
        Features selected for their strong correlation with delays include Carrier, Flight Date, and Departure Time Block.

   - Efficient Backend:
        Preprocessing pipeline ensures accurate predictions by transforming user inputs into a format compatible with the trained model.
