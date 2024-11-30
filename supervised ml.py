
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data = {
    "House Size (sq ft)": [1500, 1600, 1700, 1800, 1900, 2000, 2100],
    "Number of Rooms": [3, 3, 4, 4, 5, 5, 6],
    "Price (in $1000)": [300, 320, 340, 360, 400, 420, 460]
}


df = pd.DataFrame(data)


print("Dataset:")
print(df)


X = df[["House Size (sq ft)", "Number of Rooms"]]
y = df["Price (in $1000)"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")


plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Prices")
plt.legend()
plt.show()


new_house = [[2100, 5]]  
predicted_price = model.predict(new_house)
print(f"\nPredicted Price for a house with size {new_house[0][0]} sq ft and {new_house[0][1]} rooms: ${predicted_price[0]*1000:.2f}")
