import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class FakeNewsDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length=128):

        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        text = self.data.iloc[index]["clean_text"]
        label = self.data.iloc[index]["label"]


        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )


        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }