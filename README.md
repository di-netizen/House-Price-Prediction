## House Price Prediction

A complete machine learning pipeline to predict house prices using synthetic housing data. This project includes data generation, cleaning, exploration, model training, and price prediction using Random Forest Regression.

---

### 🔍 Features

- **Data Simulation**: Generates a dataset with key features like square footage, number of bedrooms/bathrooms, parking spots, and location.
- **Data Cleaning**: Handles missing values and applies one-hot encoding to categorical variables.
- **Exploratory Analysis**: Visualizes price distribution, correlation heatmaps, and price vs. size relationships.
- **Model Training**: Trains Linear Regression and Random Forest models, evaluates performance using MAE, MSE, and R².
- **Model Saving & Inference**: Trains a final Random Forest model on the full dataset, saves it, and performs price prediction on new inputs.

---

### 📦 Requirements

Create a `requirements.txt` file with:

```text
pandas>=1.2.0
numpy>=1.18.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
joblib>=1.0.0

