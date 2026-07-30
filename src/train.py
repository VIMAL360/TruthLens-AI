import torch
from torch.optim import AdamW
from tqdm import tqdm

from dataloader import train_loader
from model import FakeNewsClassifier


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Load model
model = FakeNewsClassifier()
model.to(device)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

# Training mode
model.train()

total_loss = 0

print("\nStarting Training...\n")

for i, batch in enumerate(tqdm(train_loader)):

    if i == 20:
        break    

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    optimizer.zero_grad()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    loss = torch.nn.functional.cross_entropy(
        outputs.logits,
        labels
    )

    loss.backward()

    optimizer.step()

    total_loss += loss.item()

average_loss = total_loss / (i + 1)

print("\nTraining Complete!")
print(f"Average Loss: {average_loss:.4f}")

torch.save(model.state_dict(), "models/truthlens_bert.pth")

print("\nModel saved successfully!")