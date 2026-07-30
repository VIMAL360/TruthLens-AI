import pandas as pd
from transformers import BertTokenizer

from dataset import FakeNewsDataset


# Load data
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")


# Add labels
fake["label"] = 0
true["label"] = 1


# Merge
news = pd.concat([fake, true], ignore_index=True)


# Simple cleaning for testing
news["clean_text"] = news["text"].str.lower()


# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)


# Create dataset
dataset = FakeNewsDataset(
    news,
    tokenizer,
    max_length=128
)


# Get first sample
sample = dataset[0]


print("="*50)
print("INPUT IDS")
print("="*50)
print(sample["input_ids"])


print("\n")
print("="*50)
print("ATTENTION MASK")
print("="*50)
print(sample["attention_mask"])


print("\n")
print("="*50)
print("LABEL")
print("="*50)
print(sample["label"])