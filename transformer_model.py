from collections import namedtuple
import numpy as np
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
import tqdm
from transformers import get_linear_schedule_with_warmup
import random
import pickle as pkl
import math
from torch import Tensor
from argparse import ArgumentParser
from generate_dataset import eps
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TokenInfo = namedtuple('TokenInfo', ['idx', 'multiplier'])
NUMBERs = list(range(1,27))
LETTERs = ['A', 'B', 'C','D', 'E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# currently using different embeddings for different tokens
# number_symbolic_rep = False

class Tokenizer:

    def __init__(self, num_numbers, num_letters, number_symbolic_rep=False):
        self.num_letters = num_letters
        self.num_numbers = num_numbers
        self.table = {}
        counter = 0

        special_tokens = ['[PAD]', '[SEP]', '[BOS]', '[EOS]']
        for i, t in enumerate(special_tokens):
            self.table[t] = i
            counter += 1

        self.table['constant'] = counter
        self.number_symbolic_rep = number_symbolic_rep
        counter += 1

        self.label_table = {}
        for i, c in enumerate(LETTERs[:num_letters]):
            self.table[c] = counter
            counter += 1
            self.label_table[c] = i

        for i, n in enumerate(NUMBERs[:num_numbers]):
            self.table[n] = counter
            counter += 1
        print(self.table)
        print(len(self.table))

    # map a token to a token info
    # in the continuous, number 2 is mapped to token "constant" and multiplier 2
    def continuous_map_token(self, token):
        if type(token) == float or type(token) == int:
            return TokenInfo(self.table['constant'], float(token))
        elif type(token) == str:
            if token == '[PAD]':
                return TokenInfo(self.table['[PAD]'], 0)
            else:
                return TokenInfo(self.table[token], 1.0)
        else:
            raise ValueError(f'Invalid token type: {type(token)}')

    # map a token to a token info
    # in the symbolic map, number 2 is mapped to token "2" and multiplier 1
    def symbolic_map_token(self, token):
        if type(token) == float:
            token = int(token + eps)
        if token == '[PAD]':
            return TokenInfo(self.table['[PAD]'], 0)
        else:
            return TokenInfo(self.table[token], 1.0)

    # a gloabl control function choosing which token map to use
    def map_token(self, token):
        if self.number_symbolic_rep:
            return self.symbolic_map_token(token)
        else:
            return self.continuous_map_token(token)

    # map a result token to a label index
    def map_label(self, label):
        return self.label_table[label]

    # input_seq is a list of tokens, of type TokenInfo
    def tokenize(self, input_seq, sol, target_len=None):

        pad_length = 0 if target_len is None else target_len - len(input_seq)
        input_seq.extend(['[PAD]'] * pad_length)

        tokens = [self.map_token(t) for t in input_seq]
        idxes = [t.idx for t in tokens]
        multipliers = [t.multiplier for t in tokens]
        label = self.map_label(sol)
        return {
            'idxes': idxes,
            'multipliers': multipliers,
            'label': label
        }


def tokenize_batch(batch, tokenizer, return_tensors='pt'):
    idxes = []
    multipliers = []
    labels = []
    target_len = max(len(b['input_seq']) for b in batch)
    for example in batch:
        tokens = tokenizer.tokenize(example['input_seq'], example['sol'], target_len)
        idxes.append(tokens['idxes'])
        multipliers.append(tokens['multipliers'])
        labels.append(tokens['label'])
    first_ex = ""
    for char in batch[0]['input_seq']:
        if char =='[PAD]':
            continue

        if type(char) == float:
            first_ex = first_ex + str(round(char, 2))
        else:
            first_ex = first_ex + str(char)
        first_ex = first_ex + " "
    first_ex = first_ex + 'SOL:' + batch[0]['sol']
    d = {
        "first_ex": [first_ex,
                     batch[0]['input_seq']],
        'idxes': torch.tensor(idxes). to(device) if return_tensors=='pt' else idxes,
        'multipliers': torch.tensor(multipliers). to(device) if return_tensors=='pt' else multipliers,
        'labels': torch.tensor(labels).to(device) if return_tensors=='pt' else labels
    }
    return d

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SimpleTransformer(nn.Module):
    def __init__(self, embedding_len, num_labels, embedding_dim, num_heads, num_layers, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.padding_idx = 0

        self.embedding_layer = nn.Embedding(embedding_len, embedding_dim, padding_idx=self.padding_idx)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.embedding_dim = embedding_dim
        self.transformer_encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(dim=self.embedding_dim, heads=num_heads,
                                    dim_head=self.embedding_dim, mlp_dim=self.embedding_dim)
            for _ in range(num_layers)])
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        self.linear_layer = nn.Linear(embedding_dim, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss_func = nn.NLLLoss()
        self.pe = PositionalEncoding(embedding_dim)
        self.to(device)

    def forward(self, tokenized_batch):
        sequences, labels, multipliers = tokenized_batch['idxes'], tokenized_batch['labels'], tokenized_batch['multipliers']
        sequences=sequences.to(device)
        src_key_padding_mask = (sequences == self.padding_idx)
        labels=labels.to(device)
        multipliers = multipliers.to(device)

        x = self.embedding_layer(sequences)
        x = self.pe(x)
        multipliers = multipliers.unsqueeze(-1).repeat(1,1,self.embedding_dim)
        x = x * multipliers
        layer_attn_weights = []
        for layer in self.transformer_encoder_layers:
            x, attn_weights = layer(x)
            layer_attn_weights.append(attn_weights)
        x = self.linear_layer(x[:, 0, :])
        logits = self.logsoftmax(x)
        return logits, labels, torch.stack(layer_attn_weights), tokenized_batch['first_ex']

    def calculate_loss(self, batch):
        logits, labels, layer_attn_weights, first_ex = self.forward(batch)
        loss = self.loss_func(logits, labels)
        return loss, layer_attn_weights, first_ex

class SingleHeadAttention(nn.Module):
    def __init__(self, input_dim, inner_dim, dropout=0.):
        super().__init__()
        self.q = nn.Linear(input_dim, inner_dim, bias=False)
        self.k = nn.Linear(input_dim, inner_dim, bias=False)
        self.v = nn.Linear(input_dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.d_k = inner_dim

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attn_weights = self.softmax(torch.matmul(Q, torch.transpose(K, -2, -1)) / np.sqrt(self.d_k))
        out = torch.matmul(attn_weights, V)

        out = self.dropout(out)
        attn_weights = torch.squeeze(attn_weights[0,:,:])
        return out, attn_weights

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.attention_heads = nn.ModuleList([
            SingleHeadAttention(dim, dim_head, dropout=dropout)
            for _ in range(heads)
        ])

    def forward(self, x):
        # modification for attention visualization
        outputs = [head(x) for head in self.attention_heads]
        out = torch.cat([o[0] for o in outputs], dim=-1)
        attn_weights = torch.stack([o[1] for o in outputs])
        return out, attn_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # modification for attention visualization
        self.layernorm = nn.LayerNorm(dim)
        self.attn_proj = nn.Linear(dim_head * heads, dim, bias=False)
        self.attn = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.feedforward = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(dropout)
        )

    def forward(self, x):
        dx, attn_weights = self.attn(self.layernorm(x))
        x = x + self.attn_proj(dx)
        x = self.feedforward(x) + x
        return x, attn_weights