import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/energy_usage_plus.csv")

X = df.drop("consumption", axis=1)
y = df["consumption"]

categorical = ["season", "district_type"]
numeric = ["temperature", "humidity", "hour", "is_weekend"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"{mape * 100:.2f}%")

results = pd.DataFrame({
    "Real": y_test.values,
    "Predicted": y_pred
})

print(results)

plt.figure(figsize=(8, 5))

sns.scatterplot(x=y_test, y=y_pred, s=80)

plt.xlabel("Real consumption")
plt.ylabel("Predicted consumption")
plt.title("Real vs Predicted consumption")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.grid(True)
plt.show()