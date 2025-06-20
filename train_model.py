import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib  # or use pickle

# Load data
df = pd.read_csv("titanic.csv")
df.columns = df.columns.str.strip()  # clean spaces

# Prepare X and y
X = df[["Height(Inches)"]]
y = df["Weight(Pounds)"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model/linear_model.pkl")

print("✅ Model trained and saved as 'linear_model.pkl'")