import pandas as pd
import matplotlib.pyplot as plt

orders = pd.read_csv("pakistani_ecommerce_dataset.csv")
students = pd.read_csv("student_lifestyle_dataset.csv")

print("DATASET 1: PAKISTANI E-COMMERCE")
print("\nQ1. Product category with most orders")
print(orders["product_category"].value_counts().head(1))

print("\nQ2. Mean order value")
print(round(orders["order_value"].mean(), 2))

print("\nQ3. City receiving most orders")
print(orders["city"].value_counts().head(1))

print("\nQ4. Most common payment method")
print(orders["payment_method"].mode()[0])

print("\nQ5. Average order value across cities")
city_values = orders.groupby("city")["order_value"].mean().sort_values(ascending=False)
print(city_values.round(2))
plt.figure(figsize=(8, 4))
plt.bar(city_values.index, city_values.values, color="#4B7098")
plt.title("Mean Order Value by City")
plt.xlabel("City")
plt.ylabel("Order Value (PKR)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("city_order_values.png", dpi=160)
plt.show()

print("\nDATASET 2: STUDENT LIFESTYLE")
print("\nQ1. Average daily study hours")
print(round(students["study_hours"].mean(), 2))

print("\nQ2. Mean sleep hours for each gender")
print(students.groupby("gender")["sleep_hours"].mean().round(2))

print("\nQ3. Screen time versus sleep hours")
relationship = students[["screen_time", "sleep_hours"]].corr().iloc[0, 1]
print("Pearson correlation:", round(relationship, 4))

print("\nQ4. Daily transport cost distribution")
plt.figure(figsize=(8, 4))
plt.hist(students["daily_transport_cost"].dropna(), bins=10,
         color="#6B9B89", edgecolor="white")
plt.title("Daily Transport Cost Distribution")
plt.xlabel("Cost (PKR)")
plt.ylabel("Student Count")
plt.tight_layout()
plt.savefig("transport_cost_distribution.png", dpi=160)
plt.show()

print("\nQ5. Internet usage by gender")
print(students.groupby("gender")["internet_usage"].mean().round(2))
