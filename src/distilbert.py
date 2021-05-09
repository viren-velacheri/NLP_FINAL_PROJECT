import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import DistilBertTokenizerFast, DistilBertModel

class DistilBert(nn.Module):
    def __init__(self, dropout):
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(DistilBert, self).__init__()

        # self.config = DistilBertConfig(dropout=dropout, hidden_dim=hidden_dim, n_layers=layers, n_heads=heads)
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-cased')
            
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

        self.linear = nn.Linear(self.distilbert.config.dim, 3)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=2)

    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)

        sequence_output = self.dropout(outputs[0])

        logits = self.linear(sequence_output)
        
        logits = self.softmax(logits)

        loss = None
        if labels != None:
            cross_entropy = CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, 3)
            active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(cross_entropy.ignore_index).type_as(labels))
            loss = cross_entropy(active_logits, active_labels)

        return (loss, logits)
