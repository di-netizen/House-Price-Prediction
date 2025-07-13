import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

locations = ['Downtown', 'Suburb', 'Rural']

data = {
    'Size_sqft': np.random.randint(600, 4000, size=n),
    'Bedrooms': np.random.randint(1, 6, size=n),
    'Bathrooms': np.random.randint(1, 4, size=n),
    'Parking': np.random.randint(0, 3, size=n),
    'Location': np.random.choice(locations, size=n)
}

df = pd.DataFrame(data)

# Simulate price based on features
df['Price'] = (
    df['Size_sqft'] * 300 +
    df['Bedrooms'] * 50000 +
    df['Bathrooms'] * 30000 +
    df['Parking'] * 15000 +
    df['Location'].map({'Downtown': 200000, 'Suburb': 100000, 'Rural': 50000}) +
    np.random.randint(-30000, 30000, size=n)  # Noise
)

# Save the dataset
df.to_csv("house_data.csv", index=False)
print("✅ Generated 'house_data.csv' with shape:", df.shape)
print(df.head())
