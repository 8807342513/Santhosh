# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load Dataset (You can replace this with your dataset)
# For demo, let's create a synthetic dataset
np.random.seed(42)
data = {
    'Area': np.random.randint(500, 5000, 100),
    'Bedrooms': np.random.randint(1, 6, 100),
    'Age': np.random.randint(0, 30, 100),
    'Price': np.random.randint(50000, 500000, 100)
}
df = pd.DataFrame(data)

# Display the dataset
print("Sample Data:\n", df.head())

# Feature selection
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Display actual vs predicted
result = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print("\nActual vs Predicted:\n", result.head())

# Plot actual vs predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.show()
