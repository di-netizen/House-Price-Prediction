import pandas as pd

# Load dataset
df = pd.read_csv("house_data.csv")

# Check for missing values
print("🔍 Missing values:\n", df.isnull().sum())

# One-hot encode location
df_encoded = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Save cleaned data
df_encoded.to_csv("house_data_cleaned.csv", index=False)
print("\n✅ Cleaned data saved as 'house_data_cleaned.csv'")
print(df_encoded.head())
