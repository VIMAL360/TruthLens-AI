import pandas as pd

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print("=" * 60)
print("FAKE NEWS DATASET")
print("=" * 60)
print(fake.head())

print("\n")

print("=" * 60)
print("REAL NEWS DATASET")
print("=" * 60)
print(true.head())