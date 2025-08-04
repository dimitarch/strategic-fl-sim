from torch import nn
from transformers import BertForSequenceClassification


class BertWithClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        self.bert.classifier = nn.Sequential(nn.Linear(768, 384), nn.Linear(384, 2))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None and input_ids.dim() == 2:
            attention_mask = (input_ids != 0).long()

        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        # Return only logits to be compatible with CrossEntropyLoss
        return outputs.logits
