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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Experiment:


    def __init__(
        self, embedding_dim, num_heads, num_layers, num_numbers,
        num_letters, num_test_data=2000, num_training_data=None, 
        num_warump_steps=5000, number_symbolic_rep=False,
        seed=0, all_experiment_folder='experiments', lr=1e-4
        ):
        self.experiment_name = f'exp_num_letters={num_letters}_num_numbers={num_numbers}_embedding_dim={embedding_dim}_num_heads={num_heads}_num_layers={num_layers}_num_training_data={num_training_data}_number_symbolic_rep={number_symbolic_rep}_seed={seed}'
        self.seed = seed

        if not os.path.exists(all_experiment_folder):
            os.mkdir(all_experiment_folder)
        self.experiment_dir = os.path.join(all_experiment_folder, self.experiment_name)
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)
        self.eval_results_path = os.path.join(self.experiment_dir, 'eval_results.pkl')

        self.num_numbers, self.num_letters, self.num_test_data = num_numbers, num_letters, num_test_data
        self.test_data = generate_in_context_dataset(num_test_data, num_numbers, num_letters, seed=self.seed + 19971017)
        self.forbidden_input_seqs = set(str(d['input_seq']) for d in self.test_data)

        self.tokenizer = Tokenizer(num_numbers, num_letters, number_symbolic_rep)
        self.embedding_dim = embedding_dim
        self.model = SimpleTransformer(len(self.tokenizer.table), num_letters, embedding_dim, num_heads, num_layers, seed=seed).to(device)
        self.num_warump_steps = num_warump_steps
        self.num_training_data = num_training_data
        self.training_data = None


        self.number_symbolic_rep = number_symbolic_rep
        self.lr=lr

    # get a batch of training data
    # if the num_training_data is not specified, then we randomly sample a batch of data
    # otherwise, we first sample a numpy random seed, and then use it to generate the data
    def get_batches(self, batch_size):
        np_seed = None
        if self.num_training_data is not None:
            # sample a numpy random seed to generate the data deterministically
            np_seed = self.seed + random.randint(0, self.num_training_data // batch_size)
        
        # generate the data
        data_batch = generate_in_context_dataset(batch_size, self.num_numbers, self.num_letters, seed=np_seed)
        # remove that data that is in the test set
        data_batch = [d for d in data_batch if str(d['input_seq']) not in self.forbidden_input_seqs]
        return tokenize_batch(data_batch, self.tokenizer)

    def train(self, steps, batch_size, eval_every=500):
        print('Training...')

        # adding optimizer here
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.num_warump_steps, num_training_steps=steps)
        
        # some variables for logging
        pbar = tqdm.trange(steps)
        loss_moving_avg = 0
        eval_results = []

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
                eval_result = self.evaluate()
                eval_result['step'] = step_idx
                eval_result['training_loss'] = loss_moving_avg

                eval_results.append(eval_result)
                acc = eval_result['acc']

                pkl.dump(eval_results, open(self.eval_results_path, 'wb'))
                print(self.experiment_name, f'Step {step_idx}: accuracy: {acc:.3f}')
                if acc > 0.99:
                    break
                self.model.train()
        torch.save(self.model.state_dict(), os.path.join(self.experiment_dir, 'final_model.pt'))

    def evaluate(self):
        bsize = 32
        labels, preds, losses = [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch_idx in range((len(self.test_data) - 1) // bsize + 1):
                data_batch = self.test_data[batch_idx * bsize: (batch_idx + 1) * bsize]
                batch = tokenize_batch(data_batch, self.tokenizer)
                logits, _ = self.model.forward(batch)

                losses.extend([-logit[label].cpu().numpy() for logit, label in zip(logits, batch['labels'])])

                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                labels.extend(batch['labels'].cpu().numpy().tolist())
        return {
            'acc': np.mean(np.array(preds) == np.array(labels)),
            'loss': np.mean(losses)
        }



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_numbers', type=int, default=26)
    parser.add_argument('--num_letters', type=int, default=26)
    parser.add_argument('--num_training_data', type=int, default=None)
    parser.add_argument('--num_steps', type=int, default=1000000)
    parser.add_argument('--number_symbolic_rep', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--all_experiment_folder', type=str, default='experiments')

    args = parser.parse_args()
    #does not work, stuck at 0.23 acc
    # experiment = Experiment(embedding_dim=args.embedding_dim, num_heads=args.num_heads,
    #                         num_layers=args.num_layers, num_numbers=args.num_numbers, num_letters=args.num_letters,
    #                         num_training_data=args.num_training_data, number_symbolic_rep=True)
    # experiment.train(steps=args.num_steps, batch_size=32)

    #does not work, stuck at 0.23 acc
    # experiment = Experiment(embedding_dim=args.embedding_dim, num_heads=args.num_heads,
    #                         num_layers=args.num_layers, num_numbers=args.num_numbers, num_letters=args.num_letters,
    #                         num_training_data=args.num_training_data, number_symbolic_rep=False)
    # experiment.train(steps=args.num_steps, batch_size=32)

    #does not work, 0.24 acc
    # experiment = Experiment(embedding_dim=256, num_heads=8,
    #                         num_layers=4, num_numbers=args.num_numbers, num_letters=args.num_letters,
    #                         num_training_data=args.num_training_data, number_symbolic_rep=False)
    # experiment.train(steps=args.num_steps, batch_size=32)

    #0.887 acc
    # experiment = Experiment(embedding_dim=512, num_heads=4,
    #                         num_layers=2, num_numbers=6, num_letters=6,
    #                         num_training_data=args.num_training_data, number_symbolic_rep=False)
    # experiment.train(steps=args.num_steps, batch_size=32)
    #0.945 acc
    # experiment = Experiment(embedding_dim=512, num_heads=4,
    #                         num_layers=2, num_numbers=6, num_letters=6,
    #                         num_training_data=args.num_training_data, number_symbolic_rep=True)
    # experiment.train(steps=args.num_steps, batch_size=32)

    #0.668 acc
    # experiment = Experiment(embedding_dim=512, num_heads=4,
    #                         num_layers=2, num_numbers=10, num_letters=10,
    #                         num_training_data=args.num_training_data, number_symbolic_rep=True)
    # experiment.train(steps=args.num_steps, batch_size=32)

    #0.999 acc
    # experiment = Experiment(embedding_dim=512, num_heads=4,
    #                         num_layers=2, num_numbers=7, num_letters=7,
    #                         num_training_data=args.num_training_data, number_symbolic_rep=True)
    # experiment.train(steps=args.num_steps, batch_size=32)


    # #0.962 acc
    # experiment = Experiment(embedding_dim=512, num_heads=4,
    #                         num_layers=2, num_numbers=8, num_letters=8,
    #                         num_training_data=args.num_training_data, number_symbolic_rep=True)
    # experiment.train(steps=args.num_steps, batch_size=32)

    experiment = Experiment(embedding_dim=512, num_heads=4,
                            num_layers=2, num_numbers=9, num_letters=9,
                            num_training_data=args.num_training_data, number_symbolic_rep=True)
    experiment.train(steps=args.num_steps, batch_size=32)
