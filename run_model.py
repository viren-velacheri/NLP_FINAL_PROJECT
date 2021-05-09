import argparse
import os
import tqdm
import torch

from src.baseline import Baseline
from src.tageval import read_train_data, read_test_data

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))

    parser.add_argument("--ex-path", type=str,
                        default=os.path.join(
                            project_root, "data", 'test', "test.nolabels.txt"),
                        help="Path to examples to tag.")
    parser.add_argument("--load-path", type=str,
                        help="Path to model to run on examples.")
    parser.add_argument("--save-out", type=str,
                        default=os.path.join(
                            project_root, "outputs", "train.out"),
                        help="Path to the tagged output file.")
    parser.add_argument("--labeled", action="store_true",
                        help="Type of training data given.")

    args = parser.parse_args()
    if os.path.exists(args.save_out):
        os.remove(args.save_out)

    exs = None
    if args.labeled:
        exs, _ = read_train_data(args.ex_path)
    else:
        exs = read_test_data(args.ex_path)

    print(len(exs), 'examples read in')

    out = open(args.save_out, 'a')

    tagged_exs = []
    
    # if args.model_type == 'stupid':
    #     for ex in exs:
    #         tagged_exs.append([(tok, 'O') for tok in ex])

    # elif args.model_type == 'baseline':
    #     model = Baseline()
    #     num_tagged = 0
    #     for ex in exs:
    #         tagged_exs.append(model.tag(ex))
    #         num_tagged += 1
    #         if num_tagged % 100 == 0:
    #             print(num_tagged)    

    # TODO: implement this (should load a model and then run on given examples and write to the out file)
    saved_state_dict = torch.load(args.load_path, map_location=None if args.cuda else lambda storage, loc: storage)

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

# TODO: make sure this works with pad/cls/sep tokens and stuff like that
def tag(self, original_toks, model_toks, outputs):
        original_toks = [i.encode('ascii', 'ignore').decode('ascii') if i != 'n\'t' else 'not' for i in original_toks]
        predictions = torch.argmax(outputs, dim=2)

        tags = [(token, self.label_list[prediction][0]) for token, prediction in zip(model_toks, predictions[0].tolist())]
        for i in range(1, len(tags)):
            if tags[i][1] == 'I' and tags[i-1][1] == 'O':
                tags[i] = (tags[i][0], 'B')
        
        tags = tags[1:-1]
        tokens = model_toks[1:-1]

        toks_processed = 0
        new_tags = []
        index = 0
        while toks_processed < len(original_toks):
            orig_tok = original_toks[toks_processed]
            tok_so_far = ''
            tag = ''
            while index < len(tags) and orig_tok != tok_so_far:
                if len(tok_so_far) == 0:
                    tag = tags[index][1]
                    

                tok_to_add = ''
                if len(tokens[index]) <= 2 or tokens[index][:2] != '##':
                    tok_to_add = tokens[index]
                else:
                    tok_to_add = tokens[index][2:]
            
                tok_so_far += tok_to_add
                index += 1
            
            new_tags.append((orig_tok, tag))
            toks_processed += 1

        for i in range(len(new_tags)):
            if new_tags[i][0] == 'RT' or new_tags[i][0] == ':' or '@' in new_tags[i][0]:
                new_tags[i] = (new_tags[i][0], 'O')

        for i in range(1, len(new_tags)):
            if new_tags[i][1] == 'I' and new_tags[i-1][1] not in 'BI':
                new_tags[i] = (new_tags[i][0], 'B')

        if new_tags[0][1] == 'I':
            new_tags[0] = (new_tags[0][0], 'B')

        return new_tags


if __name__ == '__main__':
    main()