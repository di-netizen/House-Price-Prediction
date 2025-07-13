import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("house_data_cleaned.csv")

# Price distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['Price'], bins=30, kde=True, color='skyblue')
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Price vs Size scatterplot
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='Size_sqft', y='Price', hue='Bedrooms', palette='viridis')
plt.title("Price vs Size (colored by Bedrooms)")
plt.tight_layout()
plt.show()
