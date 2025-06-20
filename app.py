import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/linear_model.pkl")

# Title
st.title("Weight Predictor (in Pounds and Kilograms)")
st.write("Enter your height (in inches) to get the predicted weight.")

# User input
height = st.number_input("Enter height (in inches):", min_value=50.0, max_value=90.0, step=0.5)

# Predict button
if st.button("Predict"):
    # Prepare input
    input_df = pd.DataFrame([[height]], columns=["Height(Inches)"])
    
    # Predict weight in pounds
    weight_pounds = model.predict(input_df)[0]
    
    # Convert to kilograms
    weight_kg = weight_pounds * 0.453592
    
    # Output
    st.success(f"Predicted Weight:")
    st.write(f"**{weight_pounds:.2f} pounds**")
    st.write(f"**{weight_kg:.2f} kilograms**")
