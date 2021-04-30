import argparse
import logging
import os
import shutil
import sys

from src.baseline import Baseline
from src.tageval import read_train_data, read_test_data

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
                        choices=["baseline"],
                        help="Model type to train.")
    parser.add_argument("--labeled", type=str, default="true",
                        choices=["true", "false"],
                        help="Type of training data given.")

    args = parser.parse_args()
    if os.path.exists(args.save_out):
        os.remove(args.save_out)

    exs = None
    if args.labeled == 'true':
        exs = read_train_data(args.ex_path)
    else:
        exs = read_test_data(args.ex_path)

    print(len(exs), 'examples read in')

    out = open(args.save_out, 'a')

    tagged_exs = []

    if args.model_type == 'baseline':
        model = Baseline()
        num_tagged = 0
        for ex in exs:
            if num_tagged == 1770:
                print(ex)
            tagged_exs.append(model.tag(ex))
            num_tagged += 1
            print(num_tagged)
    
    for tags in tagged_exs:
        for tag in tags:
            if len(tag[1]) == 0:
                out.write('O')
            else:
                out.write(tag[1])
            out.write('\n')
        out.write('\n')
    
    out.close()


if __name__ == '__main__':
    main()