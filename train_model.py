import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("train.csv")

# Features
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Target
y = data['SalePrice']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')

print("Model trained successfully!")