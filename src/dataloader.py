import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import FakeNewsDataset


# Load datasets
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Assign labels
fake_df["label"] = 0
true_df["label"] = 1

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Keep only required columns
df = df[["text", "label"]]

# Clean text
df["text"] = df["text"].str.lower()

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create datasets
train_dataset = FakeNewsDataset(
    train_texts.tolist(),
    train_labels.tolist(),
    tokenizer
)

val_dataset = FakeNewsDataset(
    val_texts.tolist(),
    val_labels.tolist(),
    tokenizer
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

print("\nDataLoaders created successfully!")

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")