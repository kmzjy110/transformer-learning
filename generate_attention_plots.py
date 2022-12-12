import numpy as np
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
import tqdm
from transformers import get_linear_schedule_with_warmup
import random
import pickle as pkl
from argparse import ArgumentParser
from transformer_model import Tokenizer, SimpleTransformer, tokenize_batch
from generate_dataset import generate_in_context_dataset
import os

from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt


model_path='experiments/exp_num_letters=26_num_numbers=26_embedding_dim=512_num_heads=1_num_layers=2_num_training_data=None_number_symbolic_rep=True_seed=2/final_model.pt'

model = SimpleTransformer(len(self.tokenizer.table), num_letters, embedding_dim, num_heads, num_layers, seed=seed).to(device)