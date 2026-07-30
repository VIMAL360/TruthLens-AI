import re
import pandas as pd

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Merge datasets
news = pd.concat([fake, true], ignore_index=True)

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+|pic\.twitter\.com/\S+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Clean every article in the text column
news["clean_text"] = news["text"].apply(clean_text)

print("=" * 60)
print("ORIGINAL TEXT")
print("=" * 60)
print(news["text"][0])

print("\n")

print("=" * 60)
print("CLEANED TEXT")
print("=" * 60)
print(news["clean_text"][0])
sample = "Hello      World"

print("Before:")
print(repr(sample))

cleaned = clean_text(sample)

print("\nAfter:")
print(repr(cleaned))