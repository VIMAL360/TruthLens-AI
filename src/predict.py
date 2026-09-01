import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# ============================================================
# Load Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)


# ============================================================
# Load BERT Model
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)


# ============================================================
# Load Trained Checkpoint
# ============================================================

checkpoint = torch.load(
    "models/truthlens_v2.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


# ============================================================
# TruthLens AI
# ============================================================

print("=" * 60)
print("          TruthLens AI - Fake News Detector")
print("=" * 60)

news = input("\nEnter a news article:\n\n")


# ============================================================
# Tokenization
# ============================================================

encoding = tokenizer(
    news,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    fake_probability = probabilities[0][0].item() * 100
    real_probability = probabilities[0][1].item() * 100

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()


# ============================================================
# Result
# ============================================================

print("\n" + "=" * 60)

print(f"Real probability : {real_probability:.2f}%")
print(f"Fake probability : {fake_probability:.2f}%")

if prediction == 0:
    print("Prediction : FAKE NEWS")
else:
    print("Prediction : REAL NEWS")

print("=" * 60)