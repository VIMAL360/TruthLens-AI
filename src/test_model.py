import torch
from transformers import BertTokenizer

from model import FakeNewsClassifier


# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)


# Sample news article
text = """
Donald Trump announced a new policy today.
The government released official information.
"""


# Tokenize input
encoding = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)


# Load model
model = FakeNewsClassifier()

# Evaluation mode
model.eval()


# Disable gradients
with torch.no_grad():

    output = model(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"]
    )


print("="*50)
print("MODEL OUTPUT LOGITS")
print("="*50)

print(output.logits)


print("\n")
print("="*50)
print("PREDICTED CLASS")
print("="*50)

prediction = torch.argmax(
    output.logits,
    dim=1
)

print(prediction)