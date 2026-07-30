import torch
from transformers import BertForSequenceClassification


class FakeNewsClassifier(torch.nn.Module):

    def __init__(self):

        super(FakeNewsClassifier, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )


    def forward(self, input_ids, attention_mask):

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return output