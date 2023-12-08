# we can input positional embedding, token embedding and token_type embedding  .. 
# and specify the attention_mask globally

# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
import copy
import numpy as np
import torch
from torch.distributions import Categorical

from torch import nn
from tools.nn_base import Network
from typing import List

from transformers import AutoConfig
from .tokenizer import Tokenizer
from .gpt2 import GPT2LMHeadModel, GPT2PreTrainedModel
from .serialize import deserialize, Code, special_tp


class SampleState:
    def __init__(
        self, codes: List[Code], past=None
    ) -> None:
        if not isinstance(codes, list):
            codes = [codes]
        self.codes = codes
        self.past = past

    @classmethod
    def new(cls, batch_size=1):
        init_code = Code()
        _, line = init_code.new_line()
        line.append("lambda", special_tp)

        code = [copy.deepcopy(init_code) for i in range(batch_size)]
        return SampleState(code)

    def prepare_inputs(self, tokenizer, device):
        def long(x):
            return torch.tensor(np.array(x), device=device, dtype=torch.long)
        tokens, linenos, col_offsets = [], [], []
        for code in self.codes:
            t, l, c = code.last_tuple
            tokens.append(t)
            linenos.append(l)
            col_offsets.append(c)
        tuples = tokenizer([tokens, linenos, col_offsets])
        return {
            'input_ids': long(tuples[0])[:, None],
            'linenos': long(tuples[1])[:, None],
            'col_offsets': long(tuples[2])[:, None],
            'past': self.past
        }

    def add_token(self, tokens, past=None):
        assert len(tokens) == len(self.codes)
        new_codes = copy.deepcopy(self.codes)
        new_codes = [deserialize(token, new_code) for token, new_code in zip(tokens, new_codes)]
        return SampleState(new_codes, past)


class CodeNet(Network, GPT2PreTrainedModel):
    is_parallelizable = False

    def __init__(self, tokenizer_or_cfg, cfg=None):
        if isinstance(tokenizer_or_cfg, Tokenizer):
            N = tokenizer_or_cfg.N()
            tokenizer = tokenizer_or_cfg
            gpt2_config = AutoConfig.from_pretrained('gpt2', vocab_size=N)
        else:
            N = tokenizer_or_cfg.vocab_size
            gpt2_config = tokenizer_or_cfg
            tokenizer = Tokenizer.from_pretrained(gpt2_config._name_or_path)

        Network.__init__(self)
        GPT2PreTrainedModel.__init__(self, config=gpt2_config)

        n_embed = gpt2_config.n_embd

        self.tokenizer = tokenizer

        self.embeddings = nn.Embedding(N, n_embed)
        self.lm = GPT2LMHeadModel(gpt2_config)
        self.lm.transformer.from_pretrained('gpt2')

    def forward(self, input_ids, linenos, col_offsets, past=None, **kwargs):
        if past:
            input_ids = input_ids[:, -1:]
            linenos = linenos[:, -1:]
            col_offsets = col_offsets[:, -1:]

        inputs_embeds = self.embeddings(input_ids)
        position_embeds = self.embeddings(linenos) + self.embeddings(col_offsets)
        return self.lm(inputs_embeds=inputs_embeds, position_embeds=position_embeds, labels=input_ids, past_key_values=past, **kwargs)

    def eval_code(self, codes: List[Code]):
        inputs = self.tokenizer.batchify(codes)
        return self.forward(**inputs)

    @torch.no_grad()
    def next(self, state: SampleState, output = None, tokens=None, sample_mode=None):
        self.lm.eval()
        if output is None:
            inputs = state.prepare_inputs(self.tokenizer, device=self.device)
            # print(self.tokenizer.decode(inputs['input_ids']), 
            # self.tokenizer.decode(inputs['linenos']), self.tokenizer.decode(inputs['col_offsets']))
            output = self(**inputs)

        if tokens is None:
            logits = output['logits'][:, -1, :]
            prob = Categorical(logits=logits)
            assert sample_mode in ['BEST', 'RANDOM']
            if sample_mode == 'BEST':
                ids = logits.argmax(axis=-1)
            elif sample_mode == 'RANDOM':
                ids = prob.sample()
            else:
                raise NotImplementedError("If token is not provided, then a sample method must be provided.")

            tokens = self.tokenizer.decode(ids)

        return state.add_token(tokens, output.past_key_values)

    @classmethod
    def from_dsl(cls, dsl):
        tokenizer = Tokenizer.from_dsl(dsl)
        return CodeNet(tokenizer)