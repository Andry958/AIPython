import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("fuel_consumption_vs_speed.csv")

df.columns = df.columns.str.strip().str.lower()

df = df.dropna()

X = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

degrees = [1, 2, 3, 4, 5, 6]

best_degree = None
best_model = None
best_poly = None
best_mse = float("inf")

for d in degrees:
    poly = PolynomialFeatures(degree=d)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Degree {d} | MSE: {mse:.4f} | MAE: {mae:.4f}")

    if mse < best_mse:
        best_mse = mse
        best_degree = d
        best_model = model
        best_poly = poly

print("\nBest degree:", best_degree)

X_sorted = np.sort(X, axis=0)
X_curve = best_poly.transform(X_sorted)
y_curve = best_model.predict(X_curve)

plt.figure(figsize=(8, 5))

plt.scatter(X, y)
plt.plot(X_sorted, y_curve, color="red")

plt.xlabel("Speed")
plt.ylabel("Fuel consumption")
plt.title("Polynomial Regression")
plt.grid()
plt.show()

test_speeds = np.array([[35], [95], [140]])
test_poly = best_poly.transform(test_speeds)
predictions = best_model.predict(test_poly)

print("\nPredictions:")
for s, p in zip(test_speeds.flatten(), predictions):
    print(f"{s} -> {p:.2f}")