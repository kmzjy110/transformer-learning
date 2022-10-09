from generate_dataset import generate_dataset
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
 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TokenInfo = namedtuple('TokenInfo', ['idx', 'multiplier'])
NUMBERs = list(range(1,27))
LETTERs = ['A', 'B', 'C','D', 'E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
eps = 0.2
# currently using different embeddings for different tokens
number_symbolic_rep = True

def generate_dataset(dataset_len = 5000, num_numbers=26, num_letters=26, save_path=None, seed=None):
    assert 2 <= num_numbers <= 26
    assert 2 <= num_letters <= 26
    numbers = NUMBERs[:num_numbers]
    letters = LETTERs[:num_letters]

    if seed is not None:
        np.random.seed(seed)

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
        prompt = generated_numbers[target_idx] + eps - np.random.rand() * 2 * eps
        input_seq.append(prompt)
        input_seq.append('[EOS]')

        data_dict = {"input_seq": input_seq, "sol": sol_letter}
        data.append(data_dict)

    if save_path is not None:
        with open(save_path, 'w') as out_file:
            json.dump(data,out_file)

    return data

# only returns the maximal character from a sequence
# turns out that you need to tune the learning rate really carefully
# it is here mostly for debugging purposes
def generate_dataset_(dataset_len = 5000, num_numbers=26, num_letters=26, save_path=None):
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

    def __init__(self, num_numbers, num_letters):
        self.num_letters = num_letters
        self.num_numbers = num_numbers
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
        
        for i, n in enumerate(NUMBERs[:num_numbers]):
            self.table[n] = counter
            counter += 1
        
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
        if number_symbolic_rep:
            return self.symbolic_map_token(token)
        else:
            return self.continuous_map_token(token)

    # map a result token to a label index
    def map_label(self, label):
        return self.label_table[label]

    # input_seq is a list of tokens, of type TokenInfo
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

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
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
        num_letters, num_test_data=2000, num_training_data=None, num_warump_steps=5000):
        self.experiment_name = f'exp_num_letters={num_letters}_num_numbers={num_numbers}_embedding_dim={embedding_dim}_num_heads={num_heads}_num_layers={num_layers}_num_training_data={num_training_data}'
        self.num_numbers, self.num_letters, self.num_test_data = num_numbers, num_letters, num_test_data
        self.test_data = generate_dataset(num_test_data, num_numbers, num_letters)
        self.forbidden_input_seqs = set(str(d['input_seq']) for d in self.test_data)

        self.tokenizer = Tokenizer(num_numbers, num_letters)
        self.embedding_dim = embedding_dim
        self.model = SimpleTransformer(len(self.tokenizer.table), num_letters, embedding_dim, num_heads, num_layers).to(device)
        self.num_warump_steps = num_warump_steps
        self.num_training_data = num_training_data

    # get a batch of training data
    # if the num_training_data is not specified, then we randomly sample a batch of data
    # otherwise, we first sample a numpy random seed, and then use it to generate the data
    def get_batches(self, batch_size):
        np_seed = None
        if self.num_training_data is not None:
            # sample a numpy random seed to generate the data deterministically
            np_seed = random.randint(0, self.num_training_data // batch_size)
        
        # generate the data
        data_batch = generate_dataset(batch_size, self.num_numbers, self.num_letters, seed=np_seed)
        # remove that data that is in the test set
        data_batch = [d for d in data_batch if str(d['input_seq']) not in self.forbidden_input_seqs]
        return tokenize_batch(data_batch, self.tokenizer)

    def train(self, steps, batch_size, eval_every=500):
        print('Training...')

        # adding optimizer here
        self.optimizer = AdamW(self.model.parameters(), lr=3e-5)
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.num_warump_steps, num_training_steps=steps)
        
        # some variables for logging
        pbar = tqdm.trange(steps)
        loss_moving_avg = 0
        accs = []

        # this part of the training is pretty much the same
        for step_idx in pbar:
            batch = self.get_batches(batch_size)
            loss = self.model.calculate_loss(batch)
            loss.backward()
            self.optimizer.step()

            # update learning rate with the scheduler
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            # update moving average
            if step_idx == 0:
                loss_moving_avg = loss.item()
            else:
                loss_moving_avg = 0.99 * loss_moving_avg + 0.01 * loss.item()
            pbar.set_description(f'loss: {loss_moving_avg:.3f}')

            # evaluate accuracy
            if step_idx % eval_every == 0:
                acc = self.evaluate()
                accs.append(acc)
                pkl.dump(accs, open(f'experiment_log/{self.experiment_name}_accs.pkl', 'wb'))
                print(self.experiment_name, f'Step {step_idx}: accuracy: {acc:.3f}')
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
    experiment = Experiment(embedding_dim=512, num_heads=8, num_layers=4, num_numbers=6, num_letters=6, num_warump_steps=5000, num_test_data=1000)
    experiment.train(steps=1000000, batch_size=64)
