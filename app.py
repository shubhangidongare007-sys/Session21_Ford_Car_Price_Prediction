# ==========================================
# Q1 - Import Required Libraries
# ==========================================

import streamlit as st      # Streamlit for Web App
import pandas as pd         # Pandas for Data Handling
import joblib               # Joblib to Load Saved Model

# ==========================================
# Q2 - Load Model and Preprocessing Objects
# ==========================================

model = joblib.load("LR_car.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

# ==========================================
# Q3 - Page Configuration
# ==========================================

st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)

# ==========================================
# Q4 - Title and Description
# ==========================================

st.title("🚗 Ford Car Price Predictor")

st.write("Enter the car details below to predict its selling price.")
# ==========================================
# Q5 - Numerical Input Fields
# ==========================================

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2026,
    value=2020
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=300000,
    value=50000
)

tax = st.number_input(
    "Road Tax",
    min_value=0,
    max_value=1000,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

engine_size = st.number_input(
    "Engine Size",
    min_value=0.5,
    max_value=10.0,
    value=1.5
)

# ==========================================
# Q6 - Categorical Input using Dropdown
# ==========================================

transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual", "Semi-Auto"]
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric"]
)
# ==========================================
# Q7 - Text Input and Predict Button
# ==========================================

model_name = st.text_input("Car Model Name")

predict = st.button("Predict Price")

# ==========================================
# Q8 & Q9 - Create DataFrame, Encoding,
# Scaling and Prediction
# ==========================================

if predict:

    input_data = pd.DataFrame({
        "year": [year],
        "mileage": [mileage],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size],
        "model": [model_name],
        "transmission": [transmission],
        "fuelType": [fuel_type]
    })

    # Convert categorical variables into dummy/indicator variables
    input_data = pd.get_dummies(input_data, drop_first=True)

        # Align columns with the original training features
    input_data = input_data.reindex(
            columns=encoded_columns,
            fill_value=0
        )

        # Apply feature scaling
    input_scaled = scaler.transform(input_data)

        # Make the vehicle price prediction
    prediction = model.predict(input_scaled)

        # Handle negative predictions by setting a baseline price
    final_price = prediction[0]
    if final_price < 0:
      final_price = 12500.00

        # Display the formatted predicted price output
    st.success(f"Predicted Price: £{final_price:,.2f}")
     # ==========================================
# Q10 - End of Streamlit Application
# ==========================================

st.markdown("---")
st.write("Ford Car Price Prediction Web App")
st.write("Developed using Streamlit, Pandas, Joblib and Scikit-learn")