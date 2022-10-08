from random import random
from generate_dataset import generate_dataset
from collections import namedtuple
import numpy as np
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
import tqdm
from transformers import get_linear_schedule_with_warmup
 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TokenInfo = namedtuple('TokenInfo', ['idx', 'multiplier'])
NUMBERs = list(range(1,27))
LETTERs = ['A', 'B', 'C','D', 'E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def generate_dataset_(dataset_len = 5000, num_numbers=26, num_letters=26, save_path=None):
    assert 2 <= num_numbers <= 26
    assert 2 <= num_letters <= 26
    numbers = NUMBERs[:num_numbers]
    letters = LETTERs[:num_letters]

    data = []
    for _ in range(dataset_len):
        num_incontext_examples = np.random.randint(2,num_numbers)
        input_seq = ['[BOS]']

        generated_numbers = np.random.choice(numbers, num_incontext_examples, replace=False).tolist()
        generated_letters = np.random.choice(letters, num_incontext_examples, replace=True).tolist()

        for n, l in zip(generated_numbers, generated_letters):
            input_seq.append(n)
            input_seq.append(l)
        input_seq.append('[SEP]')

        target_idx = np.random.randint(0, num_incontext_examples - 1)
        sol_letter = generated_letters[target_idx]
        prompt = generated_numbers[target_idx] + 0.2 - np.random.rand() * 0.4
        input_seq.append(prompt)
        input_seq.append('[EOS]')

        data_dict = {"input_seq": input_seq, "sol": sol_letter}
        data.append(data_dict)

    if save_path is not None:
        with open(save_path, 'w') as out_file:
            json.dump(data,out_file)

    return data


def generate_dataset(dataset_len = 5000, num_numbers=26, num_letters=26, save_path=None):
    assert 2 <= num_numbers <= 26
    assert 2 <= num_letters <= 26
    letters = LETTERs[:num_letters]

    data = []
    for _ in range(dataset_len):
        num_incontext_examples = np.random.randint(2,num_numbers)
        input_seq = ['[BOS]']

        generated_letters = np.random.choice(letters, num_incontext_examples, replace=True).tolist()

        input_seq.extend(generated_letters)
        input_seq.append('[SEP]')
        input_seq.append('[EOS]')

        sol_letter = max(generated_letters)

        data_dict = {"input_seq": input_seq, "sol": sol_letter}

        data.append(data_dict)

    if save_path is not None:
        with open(save_path, 'w') as out_file:
            json.dump(data,out_file)

    return data

class Tokenizer:

    def __init__(self, num_letters):
        self.num_letters = num_letters
        self.table = {}
        counter = 0

        special_tokens = ['[PAD]', '[SEP]', '[BOS]', '[EOS]']
        for i, t in enumerate(special_tokens):
            self.table[t] = i
            counter += 1
        
        self.table['constant'] = counter
        counter += 1

        self.label_table = {}
        for i, c in enumerate(LETTERs[:num_letters]):
            self.table[c] = counter
            counter += 1
            self.label_table[c] = i
    
    def map_token(self, token):
        if type(token) == float or type(token) == int:
            return TokenInfo(self.table['constant'], float(token))
        elif type(token) == str:
            if token == '[PAD]':
                return TokenInfo(self.table['[PAD]'], 0)
            else:
                return TokenInfo(self.table[token], 1.0)
        else:
            raise ValueError(f'Invalid token type: {type(token)}')
    
    def map_label(self, label):
        return self.label_table[label]

    def tokenize(self, input_seq, sol, target_len = None):

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
    d = {
        'idxes': idxes,
        'multipliers': multipliers,
        'labels': labels
    }
    if return_tensors == 'pt':
        d = {k: torch.tensor(v).to(device) for k, v in d.items()}
    return d

class SimpleTransformer(nn.Module):
    def __init__(self, embedding_len, num_labels, embedding_dim, num_heads, num_layers, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.padding_idx = 0

        self.embedding_layer = nn.Embedding(embedding_len, embedding_dim, padding_idx=self.padding_idx)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.embedding_dim = embedding_dim

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        self.linear_layer = nn.Linear(embedding_dim, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss_func = nn.NLLLoss()
        self.to(device)

    def forward(self, tokenized_batch):
        sequences, labels, multipliers = tokenized_batch['idxes'], tokenized_batch['labels'], tokenized_batch['multipliers']
        sequences=sequences.to(device)
        src_key_padding_mask = (sequences == self.padding_idx)
        labels=labels.to(device)
        multipliers = multipliers.to(device)

        x = self.embedding_layer(sequences)
        multipliers = multipliers.unsqueeze(-1).repeat(1,1,self.embedding_dim)
        x = x * multipliers
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.linear_layer(x[:, 0, :])
        logits = self.logsoftmax(x)
        return logits, labels

    def calculate_loss(self, batch):
        logits, labels = self.forward(batch)
        loss = self.loss_func(logits, labels)
        return loss


class Experiment:

    def __init__(
        self, embedding_dim, num_heads, num_layers, num_numbers, 
        num_letters, num_test_data=500, num_warump_steps=5000):
        self.num_numbers, self.num_letters, self.num_test_data = num_numbers, num_letters, num_test_data
        self.test_data = generate_dataset(num_test_data, num_numbers, num_letters)
        self.forbidden_input_seqs = set(str(d['input_seq']) for d in self.test_data)

        self.tokenizer = Tokenizer(num_letters)
        self.embedding_dim = embedding_dim
        self.model = SimpleTransformer(len(self.tokenizer.table), num_letters, embedding_dim, num_heads, num_layers).to(device)
        self.num_warump_steps = num_warump_steps

    def get_batches(self, batch_size):
        data_batch = generate_dataset(batch_size, self.num_numbers, self.num_letters)
        data_batch = [d for d in data_batch if str(d['input_seq']) not in self.forbidden_input_seqs]
        return tokenize_batch(data_batch, self.tokenizer)

    def train(self, steps, batch_size, eval_every=500):
        print('Training...')
        self.optimizer = AdamW(self.model.parameters(), lr=3e-5)
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=1000, num_training_steps=steps)
        pbar = tqdm.trange(steps)
        for step_idx in pbar:
            batch = self.get_batches(batch_size)
            loss = self.model.calculate_loss(batch)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            pbar.set_description(f'loss: {loss.item():.3f}')
            if step_idx % eval_every == 0:
                acc = self.evaluate()
                print(f'Step {step_idx}: accuracy: {acc:.3f}')
                self.model.train()

    def evaluate(self):
        bsize = 32
        labels, preds = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_idx in range((len(self.test_data) - 1) // bsize + 1):
                data_batch = self.test_data[batch_idx * bsize: (batch_idx + 1) * bsize]
                batch = tokenize_batch(data_batch, self.tokenizer)
                logits, _ = self.model.forward(batch)

                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                labels.extend(batch['labels'].cpu().numpy().tolist())
        return np.mean(np.array(preds) == np.array(labels))



if __name__ == '__main__':
    experiment = Experiment(embedding_dim=256, num_heads=8, num_layers=4, num_numbers=6, num_letters=26, num_warump_steps=5000, num_test_data=2000)
    experiment.train(steps=100000, batch_size=64)



        
        
