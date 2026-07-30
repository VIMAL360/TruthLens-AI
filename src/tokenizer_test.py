from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

text = "Fake news spreads quickly"

# Tokenize
tokens = tokenizer.tokenize(text)

print("="*50)
print("TOKENS")
print("="*50)
print(tokens)


# Convert tokens into IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\n")
print("="*50)
print("TOKEN IDS")
print("="*50)
print(token_ids)


# Complete encoding
encoding = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=10
)

print("\n")
print("="*50)
print("INPUT IDS")
print("="*50)
print(encoding["input_ids"])


print("\n")
print("="*50)
print("ATTENTION MASK")
print("="*50)
print(encoding["attention_mask"])