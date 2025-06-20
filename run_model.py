import pandas as pd
import joblib

# Load model
model = joblib.load("model/linear_model.pkl")

# Get user input
height = float(input("Enter height (in inches): "))

# Pass input with feature name as DataFrame
input_df = pd.DataFrame([[height]], columns=["Height(Inches)"])
predicted = model.predict(input_df)

print(f"Predicted weight for {height} inches: {predicted[0]:.2f} pounds")