import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertConfig, AdamW
from torch.utils.data import DataLoader

from src.tageval import read_train_data
from src.preprocess import preprocess

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))
        
    parser.add_argument("--ex-path", type=str,
                        default=os.path.join(
                            project_root, "data", 'train', "train.txt"),
                        help="Path to the training data.")
    parser.add_argument('--epochs', type=int,
                        default=25,
                        help='Number of epochs to train model for.')
    parser.add_argument('--batch-size', type=int,
                        default=32,
                        help='Batch size.')
    parser.add_argument('--dropout', type=float,
                        default=0,
                        help='Dropout rate.')

    args = parser.parse_args()
    texts, tags = read_train_data(args.ex_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

    dataset, tag2id, id2tag = preprocess(texts, tags, tokenizer)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DistilBert(dropout=args.dropout)
    model.to(device)
    model.train()

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for _ in tqdm(range(args.epochs), unit='epoch'):
        for batch in tqdm(train_loader, unit='batch'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

    model.eval()
    eval_loader = DataLoader(dataset)
    for batch in eval_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        print(outputs[1])
        exit()

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


if __name__ == '__main__':
    main()