import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_csv("house_data_cleaned.csv")
X = df.drop('Price', axis=1)
y = df['Price']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "house_price_model.pkl")
print("✅ Model saved as 'house_price_model.pkl'")

# Simulate a new house input
new_house = pd.DataFrame([{
    'Size_sqft': 1800,
    'Bedrooms': 3,
    'Bathrooms': 2,
    'Parking': 1,
    'Location_Rural': 0,
    'Location_Suburb': 1
}])

# Load the model
loaded_model = joblib.load("house_price_model.pkl")

# Predict
predicted_price = loaded_model.predict(new_house)[0]
print(f"🏠 Predicted House Price: ₹ {int(predicted_price):,}")
