import argparse
import logging
import os
import shutil
import sys
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.nn.functional import nll_loss
import random

from src.baseline import Baseline
from src.tageval import read_train_data, read_test_data, one_hot_label, char_label, new_label, read_labels_file
from distilbert import DistilBert

from transformers import AdamW

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))

    parser.add_argument("--ex-path", type=str,
                        default=os.path.join(
                            project_root, "data", 'train', "train.txt"),
                        help="Path to the training data.")
    parser.add_argument("--save-out", type=str,
                        default=os.path.join(
                            project_root, "outputs", "train.out"),
                        help="Path to the tagged output file.")
    parser.add_argument("--model-type", type=str, default="baseline",
                        choices=["baseline", 'stupid', 'distilbert'],
                        help="Model type to train.")
    parser.add_argument("--labeled", action="store_true",
                        help="Type of training data given.")
    parser.add_argument("--cuda", action="store_true",
                        help="Train or evaluate with GPU.")
    parser.add_argument("--label-path", type=str,
                        default=os.path.join(
                            project_root, "data", 'train', "train_labels.txt"),
                        help="Path to the example labels.")
    parser.add_argument('--epochs', type=int,
                        default=25,
                        help='Number of epochs to train model for.')

    args = parser.parse_args()
    if os.path.exists(args.save_out):
        os.remove(args.save_out)

    exs = None
    labels = None
    if args.labeled:
        exs, _ = read_train_data(args.ex_path)
        labels = read_labels_file(args.label_path)
    else:
        exs = read_test_data(args.ex_path)

    print(len(exs), 'examples read in')

    out = open(args.save_out, 'a')

    tagged_exs = []
    
    if args.model_type == 'stupid':
        for ex in exs:
            tagged_exs.append([(tok, 'O') for tok in ex])

    elif args.model_type == 'baseline':
        model = Baseline()
        num_tagged = 0
        for ex in exs:
            tagged_exs.append(model.tag(ex))
            num_tagged += 1
            if num_tagged % 100 == 0:
                print(num_tagged)
    
    elif args.model_type == 'distilbert':
        print('Training distilBERT model')
        model = DistilBert(args.cuda)
        if args.cuda:
            model.cuda()

        optimizer = AdamW(model.parameters())
        for i in tqdm(range(args.epochs), unit='epoch'):
            train_epoch(model, exs, labels, optimizer)

        
        exit()
        
        
    
    for tags in tagged_exs:
        print(tags)
        for tag in tags:
            if len(tag[1]) == 0:
                out.write('O')
            else:
                out.write(tag[1])
            out.write('\n')
        out.write('\n')
    
    out.close()

def train_epoch(model, exs, labels, optimizer):
    exs, labels = shuffle(exs, labels)
    batch_size = 32
    model.train()
    i = 0
    while i < len(exs):
        batch = model.tokenizer(exs[i:min(i + batch_size, len(exs))], add_special_tokens=False, is_split_into_words=True, padding=True, return_tensors='pt')
        output = model(**batch)
        loss = nll_loss(output, labels)
        

        i += batch_size
    
    raise NotImplementedError

def shuffle(exs, labels):
    temp = list(zip(exs, labels))
    random.shuffle(temp)
    return zip(*temp)

if __name__ == '__main__':
    main()