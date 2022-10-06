import generate_dataset
import torch
import torch.nn as nn
def build_lookup_table():
    counter = 0
    table = dict()

    for letter in generate_dataset.letters:
        table[letter] = counter
        counter+=1

    table['[PAD]'] = counter
    counter+=1

    table['[SEP]'] = counter
    counter+=1

    table['[BOS]'] = counter
    counter+=1

    table['[EOS]'] = counter
    counter+=1
    # for number in generate_dataset.numbers:
    #     table[str(number-0.1)] = counter
    #     counter+=1
    #     table[str(number)] = counter
    #     counter+=1
    #     table[str(number+0.1)] = counter
    #     counter+=1
    table['1'] = counter
    counter+=1

    return table, counter

def tokenize_data(batch, lookup_table):
    sequences = []
    labels = []
    multipliers = []
    for item in batch:
        current_sequence = item['sequence'].split(' ')[:-1]
        current_sequence_tokens = []
        current_multipliers = []

        current_sequence_tokens.append(lookup_table['[BOS]'])
        current_multipliers.append(1)

        current_sequence_tokens.extend([1 if char.isnumeric() else lookup_table[char] for char in current_sequence])
        current_multipliers.extend([float(char) if char.isnumeric() else 1 for char in current_sequence])

        current_sequence_tokens.append(lookup_table['[SEP]'])
        current_multipliers.append(1)

        current_sequence_tokens.append(1)
        current_multipliers.append(float(item['prompt']))

        current_sequence_tokens.append(lookup_table['[EOS]'])
        current_multipliers.append(1)

        sequences.append(current_sequence_tokens)
        labels.append(lookup_table[item['sol']])
        multipliers.append(current_multipliers)
    max_seq_len = max([len(seq) for seq in sequences])

    for i in range(len(sequences)):
        seq = sequences[i]
        current_seq_len = len(seq)
        seq.extend([lookup_table['[PAD]'] for k in range(max_seq_len-current_seq_len)])
        multiplier = multipliers[i]
        multiplier.extend([1 for k in range(max_seq_len-current_seq_len)])
    return torch.tensor(sequences), torch.tensor(labels), torch.tensor(multipliers)

class SimpleTransformer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lookup_table, self.embedding_len = build_lookup_table()
        self.device = device
        self.embedding_dim = 128
        self.embedding_layer = nn.Embedding(self.embedding_len, self.embedding_dim, padding_idx=self.lookup_table['[PAD]'])

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
        self.linear_layer = nn.Linear(self.embedding_dim, 26)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss_func = nn.NLLLoss()
        self.to(self.device)
    def forward(self, batch):
        sequences, labels, multipliers = tokenize_data(batch, self.lookup_table)
        sequences=sequences.to(self.device)
        labels=labels.to(self.device)
        multipliers = multipliers.to(self.device)

        x = self.embedding_layer(sequences)
        multipliers = multipliers.unsqueeze(-1).repeat(1,1,self.embedding_dim)
        x = x * multipliers
        x = self.transformer_encoder(x)
        x = torch.squeeze(x[:, 0, :])
        x = self.linear_layer(x)
        logits = self.logsoftmax(x)
        return logits, labels

    def calculate_loss(self, batch):
        logits, labels = self.forward(batch)
        loss = self.loss_func(logits, labels)
        return loss
