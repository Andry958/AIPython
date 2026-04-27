# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# # 1. Завантаження даних
# df = pd.read_csv('./internship_candidates_final_numeric.csv')

# # 2. Вхідні та цільові змінні
# # Experience,Grade,EnglishLevel,Age,EntryTestScore
# X = df[["Experience", "Grade", "EnglishLevel", "Age", "EntryTestScore"]]
# y = df['Accepted']

# # 3. Розділення
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Побудова моделі
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # 5. Передбачення
# y_pred = model.predict(X_test)

# # 6. Візуалізація
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.scatter(X_test['EntryTestScore'], X_test['Grade'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
# plt.title('Logistic Regression Predictions')
# plt.xlabel('Entry Test Score')
# plt.ylabel('English Level')    
# plt.colorbar(label='Predicted Class')
# plt.show()

# # plt.scatter(X_test['Grade'], X_test['Age'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
# # plt.title('Logistic Regression Predictions')
# # plt.xlabel('Grade')
# # plt.ylabel('Age')    
# # plt.colorbar(label='Predicted Class')
# # plt.show()

# # 7. Передбачення для нових даних
# new_data = pd.DataFrame({
#     'Experience': [2, 5],
#     'Grade': [3.5, 4.0],
#     'EnglishLevel': [2, 3],
#     'Age': [22, 25],
#     'EntryTestScore': [80, 90]
# })
# predictions = model.predict(new_data)
# print("Predictions for new data:", predictions)

# # 6. Оцінка
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#-----------------------------------2----------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import random

# 1. Завантаження даних
df = pd.read_csv('./internship_candidates_cefr_final.csv')

# 2. Вхідні та цільові змінні
# Experience,Grade,EnglishLevel,Age,EntryTestScore
X = df[["Experience", "Grade", "EnglishLevel", "Age", "EntryTestScore"]]
y = df['Accepted']

categorical_EnglishLevel = pd.get_dummies(X['EnglishLevel'], prefix='EnglishLevel') 
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['EnglishLevel'])
    ],
    remainder='passthrough'  # Пропускає числові ознаки без змін
)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 3. Розділення
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
)

# 4. Побудова моделі
model.fit(X_train, y_train)

# 5. Передбачення
y_pred = model.predict(X_test)

# 6. Візуалізація
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.scatter(X_test['Experience'], X_test['Grade'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title('Logistic Regression Predictions')
plt.xlabel('Experience')
plt.ylabel('Grade')    
plt.colorbar(label='Predicted Class')
plt.show()

# plt.scatter(X_test['Grade'], X_test['Age'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
# plt.title('Logistic Regression Predictions')
# plt.xlabel('Grade')
# plt.ylabel('Age')    
# plt.colorbar(label='Predicted Class')
# plt.show()

# 7. Передбачення для нових даних
new_data = pd.DataFrame({
    'Experience': [2, 5],
    'Grade': [3.5, 4.0],
    'EnglishLevel': [2, 3],
    'Age': [22, 25],
    'EntryTestScore': [80, 90]
})
predictions = model.predict(new_data)
print("Predictions for new data:", predictions)

# 6. Оцінка
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))