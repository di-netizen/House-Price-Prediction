import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_csv("house_data_cleaned.csv")

# Feature matrix (X) and target vector (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Confirm shapes
print("✅ Data split complete")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
