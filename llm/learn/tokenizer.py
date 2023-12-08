import torch
import os
import numpy as np
from torch import nn
from tools.nn_base import Network
from transformers import AutoProcessor, AutoFeatureExtractor, AutoTokenizer
from typing import List, Optional, Union, Dict
from .serialize import Code, extract_code, Operator


# https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/models/gpt2/modeling_gpt2.py
# https://huggingface.co/course/chapter6/2?fw=pt

MAXLINE = 100
MAXCOL = 20


def get_tokens(dsl):
    tokens = []
    tokens += [str(i) for i in range(100)]
    tokens += ["$" + str(i) for i in range(MAXLINE)] # pointer to line.
    tokens += ["L" + str(i) for i in range(MAXLINE)] # line_id
    tokens += ["C" + str(i) for i in range(MAXCOL)] # pointer_id
    tokens += ["i" + str(i) for i in range(10)] # input
    tokens += ["True", "False"]
    tokens += ["=", "<CR>", "lambda", "ret", "list", "EOF"]
    for i in dsl.operators:
        tokens.append(str(i))
    for i in dsl.types:
        tokens.append(str(i))
    return tokens



class Tokenizer:
    # very simple tokenizer
    def __init__(self, tokens) -> None:
        self.tokens = tokens
        assert len(set(self.tokens)) == len(self.tokens), "duplicated token found .."
        self.token2id = {self.tokens[i]: i for i in range(len(self.tokens))}

    @classmethod
    def from_pretrained(cls, output_dir):
        with open(os.path.join(output_dir, 'tokens.txt'), 'r') as f:
            return cls(f.readline().strip().split(' '))

    @classmethod
    def from_dsl(cls, dsl):
        tokens = get_tokens(dsl)
        return cls(tokens)

    def save_pretrained(self, output_dir):
        with open(os.path.join(output_dir, 'tokens.txt'), 'w') as f:
            f.write(' '.join(self.tokens))


    def N(self):
        return len(self.tokens)

    def __call__(self, code: Union[Code, Operator, Dict[str, str]]):
        if isinstance(code, Operator):
            return self(extract_code(code))
        if isinstance(code, Code):
            return self(code.serialize())
        if isinstance(code, str):
            if ' ' in code:
                return self(code.split(' '))
            return self.token2id[code]
        from datasets.arrow_dataset import Batch
        if isinstance(code, Batch):
            return Batch({k: self(v) for k, v in code.items()})
        if isinstance(code, dict):
            return {k: self(v) for k, v in code.items()}
            #return {k: self(v) for k, v in code.items()}
        if isinstance(code, list):
            return [self(i) for i in code]
        print(code, type(code))
        raise NotImplementedError()

    def batchify(self, codes: List[Code], should_map=True, device='cuda:0'):
        def long(x):
            return torch.tensor(x, dtype=torch.long, device=device)


        def pad(lists):
            # use the simplest way to pad .. 
            pad = self.token2id["EOF"]
            out = torch.zeros(len(lists), max(map(len, lists)) + 1, dtype=torch.long, device=device) + pad
            for idx, l in enumerate(lists):
                out[idx, :len(l)] = long(l)
            return out

        input_ids = []
        linenos = []
        col_offsets = []

        if should_map:
            codes = map(self, codes)
        for infos in codes:
            input_ids.append(infos['input_ids'])
            linenos.append(infos['linenos'])
            col_offsets.append(infos['col_offsets'])
        return {
            'input_ids': pad(input_ids),
            'linenos': pad(linenos),
            'col_offsets': pad(col_offsets),
        }

        
    def decode(self, tokens):
        return [self.tokens[i] for i in tokens]