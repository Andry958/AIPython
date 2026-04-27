import pandas as pd
import matplotlib.pyplot as plt

data = {
    "OrderID": [1001, 1002, 1003],
    "Customer": ["Alice", "Bob", "Alice"],
    "Product": ["Laptop", "Chair", "Mouse"],
    "Category": ["Electronics", "Furniture", "Electronics"],
    "Quantity": [1, 2, 3],
    "Price": [1500, 180, 25],
    "OrderDate": ["2023-06-01", "2023-06-03", "2023-06-05"]
}

df = pd.DataFrame(data)

df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["TotalAmount"] = df["Quantity"] * df["Price"]

print("dataframe:\n", df)
print("\nсумарний дохід:", df["TotalAmount"].sum())
print("середній чек:", df["TotalAmount"].mean())
print("\nзамовлення по клієнтах:\n", df["Customer"].value_counts())
print("\nзамовлення > 500:\n", df[df["TotalAmount"] > 500])
print("\nвідсортовано:\n", df.sort_values(by="OrderDate", ascending=False))

filtered = df[(df["OrderDate"] >= "2023-06-05") & (df["OrderDate"] <= "2023-06-10")]
print("\nперіод 5-10 червня:\n", filtered)

grouped = df.groupby("Category").agg({
    "Quantity": "sum",
    "TotalAmount": "sum"
})
print("\nпо категоріях:\n", grouped)

top_clients = df.groupby("Customer")["TotalAmount"].sum().sort_values(ascending=False).head(3)
print("\nтоп клієнти:\n", top_clients)


orders_per_date = df.groupby("OrderDate").size()
orders_per_date.index = pd.DatetimeIndex(orders_per_date.index).to_period("D")
orders_per_date.plot(kind="bar")

plt.title("кількість замовлень по датах")
plt.xlabel("дата")
plt.ylabel("кількість")
plt.show()

revenue_by_category = df.groupby("Category")["TotalAmount"].sum()
revenue_by_category.plot(kind="pie", autopct='%1.1f%%')
plt.title("розподіл доходів по категоріях")
plt.ylabel("")
plt.show()