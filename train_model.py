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
from run_model import tag

from src.distilbert import DistilBert

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
    parser.add_argument('--model-type', type=str, default='distilbert',
                        choices=['distilbert'], help='Model type to train.')
    parser.add_argument('--save-dir', type=str, default='', help='Directory to save model checkpoints.')

    args = parser.parse_args()
    texts, tags = read_train_data(args.ex_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

    dataset, tag2id, id2tag = preprocess(texts, tags, tokenizer)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = None
    if args.model_type == 'distilbert':
        model = DistilBert(dropout=args.dropout)

    model.to(device)
    model.train()

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in tqdm(range(args.epochs), unit='epoch'):
        for batch in tqdm(train_loader, unit='batch'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

        if epoch + 1 % 5 == 0:
            torch.save(model, args.save_dir + '/' + args.model_type + '_epoch_' + str(epoch + 1))

    model.eval()

    # TODO: evaluate on training and validation sets
    eval_loader = DataLoader(dataset)
    i = 0
    for batch in eval_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        
        original_toks = texts[i]
