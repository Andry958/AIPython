from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = load_wine()
X = data.data
y = data.target

print("Original features:\n", X[:5])
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Normalized features:\n", X[:5])

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build a simple NN for numeric data
model = tf.keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(X.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(3, activation="softmax")  # 3 класи
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=25, batch_size=16, validation_split=0.2)

model.evaluate(X_test, y_test)
# 1,14.23,1.71,2.43,15.6,127,2.8,3.06,.28,2.29,5.64,1.04,3.92,1065
sample = np.array([[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]])
sample = scaler.transform(sample)  # Normalize
pred_logits = model.predict(sample)

print("Logits:", pred_logits)

pred_class = np.argmax(pred_logits, axis=1)
print("Predicted class:", pred_class)

#model.summary()

# # Example numeric training data
# # X: 100 rows with 2 features
# X = np.random.rand(100, 2).astype("float32")

# # y: target values
# y = (X[:, 0] * 3 + X[:, 1] * 2 + 1).astype("float32")  # simple formula

# # Train model
# model.fit(X, y, epochs=50, batch_size=25)

# # Make a prediction
# example = np.array([[0.3, 0.8]])
# pred = model.predict(example)

# print("Prediction:", pred)