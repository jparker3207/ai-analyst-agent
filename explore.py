import pandas as pd

df = pd.read_csv("serverlife_shifts.csv")

print("--- First 5 rows ---")
print(df.head())

print("\n--- Column info ---")
print(df.info())

print("\n--- Quick stats on the numbers ---")
print(df.describe())