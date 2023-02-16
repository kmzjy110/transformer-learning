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
from collections import defaultdict
from torch import linalg as LA

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def model_l2_norm(model):
    with torch.no_grad():
        l2_norm = 0
        for name, param in model.named_parameters():
            l2_norm += LA.norm(param) ** 2
        return l2_norm.detach().cpu().numpy()


def evaluate_model_on_data(model, tokenizer, test_data, num_data_to_plot_attention=32, bsize=32):
    labels, preds, losses = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx in range((len(test_data) - 1) // bsize + 1):
            data_batch = test_data[batch_idx * bsize: (batch_idx + 1) * bsize]
            batch = tokenize_batch(data_batch, tokenizer)
            logits, _, _, _ = model.forward(batch)

            losses.extend([-logit[label].cpu().numpy() for logit, label in zip(logits, batch['labels'])])

            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(batch['labels'].cpu().numpy().tolist())
    
        data_batch = test_data[:num_data_to_plot_attention]
        batch = tokenize_batch(data_batch, tokenizer)
        _, _, attention, _ = model.forward(batch)

    return {
        'acc': float(np.mean(np.array(preds) == np.array(labels))),
        'loss': float(np.mean(losses)),
        'preds': preds,
        'labels': labels,
        'attention': attention.cpu().detach().numpy().tolist(),
        'l2_norm': float(model_l2_norm(model))
    }


def get_averaged_model(model_1, model_2):
    avg_model = model_1.copy_self()
    avg_model_state_dict = avg_model.state_dict()
    model_1_state_dict = model_1.state_dict()
    model_2_state_dict = model_2.state_dict()
    for key in avg_model_state_dict.keys():
        avg_model_state_dict[key] = (model_1_state_dict[key] + model_2_state_dict[key]) / 2
    avg_model.load_state_dict(avg_model_state_dict)
    return avg_model

def sample_seeds(num_samples):
    xs = set()
    while len(xs) < num_samples:
        xs.add(np.random.randint(0, 2**32 - 1))
    return list(xs)

class Experiment:


    def __init__(
        self, embedding_dim, num_heads, num_layers, num_numbers,
        num_letters, num_test_data=2000, num_training_data=None, 
        num_warump_steps=5000, number_symbolic_rep=False,
        lr=1e-4, max_steps=200000, batch_size=32,
        model_seed=0, data_seed=0, all_experiment_folder='experiments',
        acc_change_vis_threshold=0.1, save_attn_every_num_steps=10000, 
        produce_plots=False, plot_attn_seq_len=10,
        save_model_every_num_steps = None
        ):
        self.experiment_name = f'exp_num_letters={num_letters}_num_numbers={num_numbers}_embedding_dim={embedding_dim}_num_heads={num_heads}_num_layers={num_layers}_num_training_data={num_training_data}_number_symbolic_rep={number_symbolic_rep}_model_seed={model_seed}_data_seed={data_seed}_lr={lr}_batch_size={batch_size}_max_steps={max_steps}_num_warump_steps={num_warump_steps}'
        self.model_seed, self.data_seed = model_seed, data_seed

        if not os.path.exists(all_experiment_folder):
            os.mkdir(all_experiment_folder)

        self.experiment_dir = os.path.join(all_experiment_folder, self.experiment_name)
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)
        self.eval_results_path = os.path.join(self.experiment_dir, 'eval_results.jsonl')
        if os.path.exists(self.eval_results_path):
            os.remove(self.eval_results_path)

        # model parameters
        self.number_symbolic_rep = number_symbolic_rep
        self.num_numbers, self.num_letters = num_numbers, num_letters
        self.num_heads, self.num_layers = num_heads, num_layers
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_numbers, num_letters, number_symbolic_rep)
        self.model = SimpleTransformer(len(self.tokenizer.table), num_letters, embedding_dim, num_heads, num_layers, seed=self.model_seed).to(device)
        final_model_path = os.path.join(self.experiment_dir, 'final_model.pt')
        if os.path.exists(final_model_path):
            print('loading final model from path ', final_model_path, '...')
            self.final_model = self.model.copy_self()
            self.final_model.load_state_dict(torch.load(final_model_path))
        else:
            self.final_model = None

        # training parameters
        self.num_warump_steps = num_warump_steps
        self.lr=lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.save_model_every_num_steps = save_model_every_num_steps

        # data parameters
        self.test_data = generate_in_context_dataset(num_test_data, num_numbers, num_letters, seed=0)
        self.forbidden_input_seqs = set(str(d['input_seq']) for d in self.test_data)
        self.num_training_data = num_training_data
        np.random.seed(self.data_seed)
        if self.num_training_data is not None:
            self.training_seeds = sample_seeds(num_training_data // batch_size)# np.random.randint(0, 2**32 - 1, size=)
        else:
            self.training_seeds = sample_seeds(max_steps)#np.random.randint(0, 2**32 - 1, size=max_steps)
        
        # plotting the attention
        self.acc_change_vis_threshold = acc_change_vis_threshold
        self.save_attn_every_num_steps = save_attn_every_num_steps
        self.produce_plots = produce_plots
        self.plot_attn_data_gen_seq_len = plot_attn_seq_len
        self.attn_matrix_plot_seq_len = self.plot_attn_data_gen_seq_len * 2 + 4

        experiment_info = {
            'experiment_name': self.experiment_name,
            'model_seed': self.model_seed,
            'data_seed': self.data_seed,
            'num_numbers': self.num_numbers,
            'num_letters': self.num_letters,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'embedding_dim': self.embedding_dim,
            'number_symbolic_rep': self.number_symbolic_rep,
            'num_warump_steps': self.num_warump_steps,
            'lr': self.lr,
            'max_steps': self.max_steps,
            'batch_size': self.batch_size,
            'num_training_data': self.num_training_data,
            'save_model_every_num_steps': self.save_model_every_num_steps,
            'save_attn_every_num_steps': self.save_attn_every_num_steps,
            'acc_change_vis_threshold': self.acc_change_vis_threshold,
            'produce_plots': self.produce_plots,
            'plot_attn_seq_len': self.plot_attn_data_gen_seq_len,
            'attn_matrix_plot_seq_len': self.attn_matrix_plot_seq_len,
            'training_seeds': self.training_seeds,
            'test_data': self.test_data,
            'max_steps': self.max_steps
        }
        with open(os.path.join(self.experiment_dir, 'experiment_info.json'), 'w') as f:
            f.write(json.dumps(experiment_info, indent=4))


    # get a batch of training data
    # if the num_training_data is not specified, then we randomly sample a batch of data
    # otherwise, we first sample a numpy random seed, and then use it to generate the data
    def get_batches(self, batch_size, seed, first_ex_to_plot=False, fix_len=None):
        data_batch = generate_in_context_dataset(batch_size, self.num_numbers, self.num_letters, seed=seed, fix_len=fix_len)
        # remove that data that is in the test set
        data_batch = [d for d in data_batch if str(d['input_seq']) not in self.forbidden_input_seqs]
        return tokenize_batch(data_batch, self.tokenizer)

    def train(self, batch_size, eval_every=500, final_model=None, start_idx=0, optimizer=None, lr_scheduler=None, model=None):
        print('Training...')

        # adding optimizer here
        if optimizer is not None:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
            self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.num_warump_steps, num_training_steps=self.max_steps)
        
        # some variables for logging
        pbar = tqdm.trange(start_idx, self.max_steps)
        loss_moving_avg = 0
        eval_results = []

        # this part of the training is pretty much the same
        prev_acc, plots_to_save, rolling_plots, acc_one_saved = 0, [], [], False

        for step_idx in pbar:
            # get batch based on the seed
            training_seed = self.training_seeds[step_idx % len(self.training_seeds)]
            batch = self.get_batches(batch_size, training_seed)

            # train
            loss, layer_attn_weights, first_ex = self.model.calculate_loss(batch)
            loss.backward()
            self.optimizer.step()
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
                eval_result = evaluate_model_on_data(self.model, self.tokenizer, self.test_data)
                eval_result['step'] = step_idx
                eval_result['training_loss'] = float(loss_moving_avg)

                if self.final_model is not None:
                    average_model = get_averaged_model(self.model, self.final_model)
                    lc_result = evaluate_model_on_data(average_model, self.tokenizer, self.test_data)
                    for k in lc_result:
                        eval_result[f'linear_connectivity_{k}'] = lc_result[k]

                eval_results.append(eval_result)
                acc = eval_result['acc']

                with open(self.eval_results_path, 'a') as f:
                    f.write(json.dumps(eval_result) + '\n')
                print(self.experiment_name, f'Step {step_idx}: accuracy: {acc:.3f}')

                self.model.train()

            if self.save_model_every_num_steps:
                if step_idx % self.save_model_every_num_steps == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.experiment_dir, 'model_step_' + str(step_idx) + '.pt'))
                    print('model saved: ','model_step_' + str(step_idx) + '.pt')
        torch.save(self.model.state_dict(), os.path.join(self.experiment_dir, 'final_model.pt'))

    def evaluate(self, model=None):
        bsize = 32
        labels, preds, losses = [], [], []
        if model is None:
            model = self.model
        model.eval()
        with torch.no_grad():
            for batch_idx in range((len(self.test_data) - 1) // bsize + 1):
                data_batch = self.test_data[batch_idx * bsize: (batch_idx + 1) * bsize]
                batch = tokenize_batch(data_batch, self.tokenizer)
                logits, _, _, _ = model.forward(batch)

                losses.extend([-logit[label].cpu().numpy() for logit, label in zip(logits, batch['labels'])])

                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                labels.extend(batch['labels'].cpu().numpy().tolist())
        
            _, _, attention, _ = model.forward(self.data_to_plot_attention)

        return {
            'acc': float(np.mean(np.array(preds) == np.array(labels))),
            'loss': float(np.mean(losses)),
            'preds': preds,
            'labels': labels,
            'attention': attention.cpu().detach().numpy()
        }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_numbers', type=int, default=26)
    parser.add_argument('--num_letters', type=int, default=26)
    parser.add_argument('--num_training_data', type=int, default=None)
    parser.add_argument('--num_steps', type=int, default=2000000)
    parser.add_argument('--number_symbolic_rep', action='store_true')
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('--model_seed', type=int, default=0)
    parser.add_argument('--all_experiment_folder', type=str, default='experiments')
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    experiment = Experiment(embedding_dim=args.embedding_dim, num_heads=args.num_heads,
                            num_layers=args.num_layers, num_numbers=26, num_letters=26,
                            num_training_data=args.num_training_data, number_symbolic_rep=True, data_seed=args.data_seed, save_model_every_num_steps=None, model_seed=args.model_seed, all_experiment_folder=args.all_experiment_folder, max_steps=args.num_steps)
    experiment.train(batch_size=args.batch_size)
