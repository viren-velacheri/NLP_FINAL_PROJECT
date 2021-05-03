from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

class Baseline:
    def __init__(self):
        self.model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        self.label_list = [
            "O",       # Outside of a named entity
            "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
            "I-MISC",  # Miscellaneous entity
            "B-PER",   # Beginning of a person's name right after another person's name
            "I-PER",   # Person's name
            "B-ORG",   # Beginning of an organisation right after another organisation
            "I-ORG",   # Organisation
            "B-LOC",   # Beginning of a location right after another location
            "I-LOC"    # Location
        ]

    def tag(self, original_toks):
        original_toks = [i.encode('ascii', 'ignore').decode('ascii') if i != 'n\'t' else 'not' for i in original_toks]
        tokens = original_toks
        sequence = self.tokenizer.convert_tokens_to_string(tokens)

        tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(sequence)))
        inputs = self.tokenizer.encode(sequence, return_tensors="pt")

        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)

        tags = [(token, self.label_list[prediction][0]) for token, prediction in zip(tokens, predictions[0].tolist())]
        for i in range(1, len(tags)):
            if tags[i][1] == 'I' and tags[i-1][1] == 'O':
                tags[i] = (tags[i][0], 'B')
        
        tags = tags[1:-1]
        tokens = tokens[1:-1]

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
            if new_tags[i][1] == 'I' and new_tags[i-1][1] == 'O':
                new_tags[i] = (new_tags[i][0], 'B')

        if new_tags[0][1] == 'I':
            new_tags[0] = (new_tags[0][0], 'B')

        return new_tags
            


