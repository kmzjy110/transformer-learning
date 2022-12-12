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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Experiment:


    def __init__(
        self, embedding_dim, num_heads, num_layers, num_numbers,
        num_letters, num_test_data=2000, num_training_data=None, 
        num_warump_steps=5000, number_symbolic_rep=False,
        seed=0, all_experiment_folder='experiments', lr=1e-4, acc_change_vis_threshold=0.1, save_attn_every_num_steps=10000, produce_plots=False, plot_attn_seq_len=10
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
        self.num_heads, self.num_layers = num_heads, num_layers
        self.test_data = generate_in_context_dataset(num_test_data, num_numbers, num_letters, seed=self.seed + 19971017)

        self.forbidden_input_seqs = set(str(d['input_seq']) for d in self.test_data)
        self.tokenizer = Tokenizer(num_numbers, num_letters, number_symbolic_rep)
        self.embedding_dim = embedding_dim
        self.model = SimpleTransformer(len(self.tokenizer.table), num_letters, embedding_dim, num_heads, num_layers, seed=seed).to(device)
        self.num_warump_steps = num_warump_steps
        self.num_training_data = num_training_data
        self.training_data = None
        self.acc_change_vis_threshold = acc_change_vis_threshold
        self.save_attn_every_num_steps = save_attn_every_num_steps

        self.number_symbolic_rep = number_symbolic_rep
        self.lr=lr
        self.produce_plots = produce_plots
        self.plot_attn_data_gen_seq_len = plot_attn_seq_len
        self.attn_matrix_plot_seq_len = self.plot_attn_data_gen_seq_len * 2 + 4

        plot_batch_seed=self.seed+2000
        seed_add=0
        self.plot_batch = generate_in_context_dataset(1, self.num_numbers, self.num_letters, seed=self.seed+2000, fix_len=self.plot_attn_data_gen_seq_len)
        while str(self.plot_batch[0]['input_seq']) in self.forbidden_input_seqs:
            self.plot_batch = generate_in_context_dataset(1, self.num_numbers, self.num_letters, seed=plot_batch_seed,
                                                     fix_len=self.plot_attn_data_gen_seq_len)
            seed_add+=1
    # get a batch of training data
    # if the num_training_data is not specified, then we randomly sample a batch of data
    # otherwise, we first sample a numpy random seed, and then use it to generate the data
    def get_batches(self, batch_size, first_ex_to_plot=False, fix_len=None):
        np_seed = None
        if self.num_training_data is not None:
            # sample a numpy random seed to generate the data deterministically
            np_seed = self.seed + random.randint(0, self.num_training_data // batch_size)
        
        # generate the data
        if first_ex_to_plot:
            plot_batch = self.plot_batch
            data_batch = generate_in_context_dataset(batch_size-1, self.num_numbers, self.num_letters, seed=np_seed)

            data_batch = plot_batch + [d for d in data_batch if str(d['input_seq']) not in self.forbidden_input_seqs]
        else:
            if fix_len is None:
                data_batch = generate_in_context_dataset(batch_size, self.num_numbers, self.num_letters, seed=np_seed)
            else:
                data_batch = generate_in_context_dataset(batch_size, self.num_numbers, self.num_letters, seed=np_seed, fix_len=fix_len)
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
        prev_acc = 0
        plots_to_save = []
        rolling_plots = []
        acc_one_saved=False
        for step_idx in pbar:
            batch = self.get_batches(batch_size)
            loss, layer_attn_weights, first_ex = self.model.calculate_loss(batch)
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

                if self.produce_plots:
                    self.model.eval()
                    batch = self.get_batches(batch_size, first_ex_to_plot=True)
                    _, _, layer_attn_weights, first_ex = self.model.forward(batch)
                    self.optimizer.zero_grad()
                    current_plots = []
                    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
                    for layer in range(layer_attn_weights.shape[0]):

                        for head in range(layer_attn_weights.shape[1]):
                            # current_ax = axs[(head//2)][head%2]
                            # current_ax = axs[head]
                            current_ax = axs[layer]
                            ax = sns.heatmap(torch.squeeze(layer_attn_weights[layer, head, :self.attn_matrix_plot_seq_len, :self.attn_matrix_plot_seq_len]).cpu().tolist(),
                                             linewidth=0.5, annot=True, annot_kws={"fontsize": 1}, xticklabels=first_ex[1][:self.attn_matrix_plot_seq_len],
                                             yticklabels=first_ex[1][:self.attn_matrix_plot_seq_len], ax=current_ax, square=True, cbar_kws={"shrink":0.25})
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=4)
                            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4)
                            ax.invert_yaxis()
                            ax.set_title(f"Step {step_idx} attention weights at layer {layer} head {head}, with accuracy {acc:.3f}" +
                                         '\n' + "Sequence:" + first_ex[0], size=5)
                        # current_plots.append(axs[0][0])
                        current_plots.append(axs[0])

                    rolling_plots.append(current_plots)

                    if acc - prev_acc > self.acc_change_vis_threshold or (acc>0.99 and not acc_one_saved) \
                            or (self.save_attn_every_num_steps!=0 and step_idx % self.save_attn_every_num_steps==0):
                        for plot_set in rolling_plots:
                            plots_to_save.extend(plot_set)
                        rolling_plots = []
                        self.save_plots(plots_to_save)
                        if acc>0.99:
                            acc_one_saved=True

                    if len(rolling_plots) == 2:
                        plots_to_close = rolling_plots[0]
                        for plot in plots_to_close:
                            plot.cla()
                        rolling_plots = rolling_plots[1:]

                    prev_acc = acc
                    self.get_corr_of_attn_matrices(layer_attn_weights)

                # if acc > 0.99:
                #     break
                self.model.train()
        torch.save(self.model.state_dict(), os.path.join(self.experiment_dir, 'final_model.pt'))
    def save_plots(self, plots_to_save, special_plot_filename=None):
        if special_plot_filename is None:
            special_plot_filename = "plots"
        pdf_path = os.path.join(self.experiment_dir, special_plot_filename + ".pdf")
        with PdfPages(pdf_path) as pp:

            for plot in plots_to_save:
                pp.savefig(plot.figure)
        print("Saved plots:" + pdf_path)

    def get_corr_of_attn_matrices(self, attn_weights):
        #attn_weights in [layer, num_heads, seq_len, seq_len]
        layer_num, num_heads, seq_len, _ = attn_weights.shape
        corrs = []
        for layer_1 in range(layer_num):
            for head_1 in range(num_heads):
                for layer_2 in range(layer_num):
                    for head_2 in range(num_heads):
                        if layer_1==layer_2 and head_1==head_2:
                            continue
                        attn_vec_1 = torch.flatten(attn_weights[layer_1, head_1, :self.attn_matrix_plot_seq_len, :self.attn_matrix_plot_seq_len]).cpu().detach().numpy()
                        attn_vec_2 = torch.flatten(attn_weights[layer_2, head_2, :self.attn_matrix_plot_seq_len, :self.attn_matrix_plot_seq_len]).cpu().detach().numpy()
                        corr = np.corrcoef(attn_vec_1,attn_vec_2)[0][1]
                        corrs.append(corr)
        print("correlation of attention matrices:")
        print(corrs)
        return corrs, np.mean(corrs)


    def evaluate(self):
        bsize = 32
        labels, preds, losses = [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch_idx in range((len(self.test_data) - 1) // bsize + 1):
                data_batch = self.test_data[batch_idx * bsize: (batch_idx + 1) * bsize]
                batch = tokenize_batch(data_batch, self.tokenizer)
                logits, _, _, _ = self.model.forward(batch)

                losses.extend([-logit[label].cpu().numpy() for logit, label in zip(logits, batch['labels'])])

                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                labels.extend(batch['labels'].cpu().numpy().tolist())
        return {
            'acc': np.mean(np.array(preds) == np.array(labels)),
            'loss': np.mean(losses)
        }

    def generate_model_attention_plots(self, model_load_path=None):
        if model_load_path is not None:
            self.model.load_state_dict(torch.load(model_load_path))
        self.model.eval()
        current_plots = []
        for i in range(32):

            batch = self.get_batches(1, fix_len=self.plot_attn_data_gen_seq_len)
            _, _, layer_attn_weights, first_ex = self.model.forward(batch)

            fig, axs = plt.subplots(1, 2, figsize=(12, 12))
            for layer in range(layer_attn_weights.shape[0]):

                for head in range(layer_attn_weights.shape[1]):
                    # current_ax = axs[(head//2)][head%2]
                    # current_ax = axs[head]
                    current_ax = axs[layer]
                    ax = sns.heatmap(torch.squeeze(layer_attn_weights[layer, head, :self.attn_matrix_plot_seq_len,
                                                   :self.attn_matrix_plot_seq_len]).cpu().tolist(),
                                     linewidth=0.5, annot=True, annot_kws={"fontsize": 1},
                                     xticklabels=first_ex[1][:self.attn_matrix_plot_seq_len],
                                     yticklabels=first_ex[1][:self.attn_matrix_plot_seq_len], ax=current_ax,
                                     square=True, cbar_kws={"shrink": 0.25})
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=4)
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4)
                    ax.invert_yaxis()
                    ax.set_title(
                        f"Attention weights at layer {layer} head {head}" +
                        '\n' + "Sequence:" + first_ex[0], size=5)
                # current_plots.append(axs[0][0])
            current_plots.append(axs[0])
        self.save_plots(current_plots, "32_attn_plots")

    def compare_attention_heads(self, model_load_paths, save_plots=False):
        models = []

        for path in model_load_paths:

            model = SimpleTransformer(len(self.tokenizer.table), self.num_letters, self.embedding_dim, self.num_heads, self.num_layers).to(device)
            model.load_state_dict(torch.load(path))
            models.append(model)
        attn_metric_results = defaultdict(list)

        data_gen_len = 10
        seq_len=24
        plots = [[] for _ in range(len(models))]
        for data_index in range(100):
            batch = self.get_batches(1,fix_len=data_gen_len)

            current_attention_weights =[]
            #refactor

            for model_index, model in enumerate(models):
                _,_,layer_attn_weights, first_ex = model.forward(batch)
                current_attention_weights.append(layer_attn_weights)
                if save_plots:
                    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
                    for layer in range(layer_attn_weights.shape[0]):

                        for head in range(layer_attn_weights.shape[1]):
                            # current_ax = axs[(head//2)][head%2]
                            # current_ax = axs[head]
                            current_ax = axs[layer]
                            ax = sns.heatmap(torch.squeeze(layer_attn_weights[layer, head, :self.attn_matrix_plot_seq_len,
                                                           :self.attn_matrix_plot_seq_len]).cpu().tolist(),
                                             linewidth=0.5, annot=True, annot_kws={"fontsize": 1},
                                             xticklabels=first_ex[1][:self.attn_matrix_plot_seq_len],
                                             yticklabels=first_ex[1][:self.attn_matrix_plot_seq_len], ax=current_ax,
                                             square=True, cbar_kws={"shrink": 0.25})
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=4)
                            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4)
                            ax.invert_yaxis()
                            ax.set_title(
                                f"Attention weights at layer {layer} head {head}" +
                                '\n' + "Sequence:" + first_ex[0], size=5)
                        # current_plots.append(axs[0][0])
                    plots[model_index].append(axs[0])


            for i in range(len(current_attention_weights)-1):
                for j in range(i+1, len(current_attention_weights)):
                    for layer in range(current_attention_weights[0].shape[0]):
                        for head in range(current_attention_weights[0].shape[1]):
                            i_weights = current_attention_weights[i][layer, head, :seq_len, :seq_len]
                            j_weights = current_attention_weights[j][layer, head, :seq_len, :seq_len]
                            max_i_tokens = i_weights.argmax(dim=-1)
                            max_j_tokens = j_weights.argmax(dim=-1)
                            diff = torch.abs(max_i_tokens - max_j_tokens).float()
                            num_diff = torch.sum(diff!=0).float()
                            avg_diff = torch.mean(diff) * seq_len / num_diff
                            attn_metric_results[(i, layer,head, 0)].append(num_diff)
                            attn_metric_results[(i,layer,head,1)].append(avg_diff)
                            attn_metric_results[(j,layer,head,0)].append(num_diff)
                            attn_metric_results[(j,layer,head,1)].append(avg_diff)
                            if layer==1:
                                first_i_token = max_i_tokens[0]
                                first_j_token = max_j_tokens[0]
                                first_token_diff = torch.abs(first_i_token-first_j_token).float()
                                print("data_index", data_index)
                                print("first_i_token", first_i_token)
                                print("first_j_token", first_j_token)
                                print("diff", first_token_diff)
                                print()
                                attn_metric_results[(i,layer,head,2)].append(first_token_diff)
                                attn_metric_results[(j, layer, head, 2)].append(first_token_diff)
        agg_metrics = defaultdict(float)

        for i in range(len(models)):
            for layer in range(self.num_layers):
                for head in range(self.num_heads):
                    agg_metrics[(i,layer,head,0)] = torch.mean(torch.tensor(attn_metric_results[(i,layer,head,0)]))
                    agg_metrics[(i, layer, head, 1)] = torch.mean(torch.tensor(attn_metric_results[(i, layer, head, 1)]))
                    if layer==1:
                        agg_metrics[(i,layer,head,2)] = torch.mean(torch.tensor(attn_metric_results[(i,layer,head,2)]))
        print(agg_metrics)
        if save_plots:
            for i in range(len(models)):
                self.save_plots(plots[i], "attn_plots_model_"+str(i))

    def pred_1_head_model(self, model_load_path):
        model = SimpleTransformer(len(self.tokenizer.table), self.num_letters, self.embedding_dim, self.num_heads,
                                  self.num_layers).to(device)
        model.load_state_dict(torch.load(model_load_path))
        layer_0_agg_pred_acc=0
        layer_1_agg_pred_acc=0
        layer_1_first_token_agrees = 0
        for data_index in range(100):
            batch = self.get_batches(1,fix_len=10)
            _, _, layer_attn_weights, first_ex = model.forward(batch)
            layer_0_max_tokens = layer_attn_weights[0,0,:,:].argmax(dim=-1)
            layer_0_agrees = 0
            layer_0_agrees += layer_0_max_tokens[0] ==22
            for i in range(2,6):
                layer_0_agrees +=layer_0_max_tokens[i] ==1 or layer_0_max_tokens[i]==2
            for i in range(6,23):
                layer_0_agrees += layer_0_max_tokens[i] == i-4 or layer_0_max_tokens[i] == i-3

            layer_0_agrees += layer_0_max_tokens[23] ==20 or layer_0_max_tokens[23] ==21
            layer_0_pred_acc = layer_0_agrees.float()/23
            layer_0_agg_pred_acc+=layer_0_pred_acc

            layer_1_max_tokens = layer_attn_weights[1,0,:,:].argmax(dim=-1)

            sol_token = batch['first_ex'][0][-1]
            sol_index = batch['first_ex'][1].index(sol_token)
            layer_1_agrees = 0

            layer_1_agrees +=layer_1_max_tokens[0] == sol_index+3
            if not (layer_1_max_tokens[0] == sol_index+3):
                print(0)
                print(sol_index)
                print(layer_1_max_tokens[0])
                print()
            else:
                layer_1_first_token_agrees+=1
            for i in range(2,6):
                layer_1_agrees += layer_1_max_tokens[i] ==3 or layer_1_max_tokens[i] == 4 or layer_1_max_tokens[i]==5

            pred_indices = [7,9,11,13,15,17,19,21]
            for i in pred_indices:
                layer_1_agrees += layer_1_max_tokens[i]==i

            layer_1_pred_acc = layer_1_agrees.float()/13
            layer_1_agg_pred_acc +=layer_1_pred_acc
        print(layer_0_agg_pred_acc/100)
        print(layer_1_agg_pred_acc/100)
        print(layer_1_first_token_agrees)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_numbers', type=int, default=26)
    parser.add_argument('--num_letters', type=int, default=26)
    parser.add_argument('--num_training_data', type=int, default=None)
    parser.add_argument('--num_steps', type=int, default=2000000)
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

    #0.994
    # experiment = Experiment(embedding_dim=512, num_heads=4,
    #                         num_layers=2, num_numbers=9, num_letters=9,
    #                         num_training_data=args.num_training_data, number_symbolic_rep=True)
    # experiment.train(steps=args.num_steps, batch_size=32)
    #0.44?
    experiment = Experiment(embedding_dim=512, num_heads=1,
                            num_layers=2, num_numbers=26, num_letters=26,
                            num_training_data=args.num_training_data, number_symbolic_rep=True, seed=8)
    # experiment.generate_model_attention_plots("experiments/exp_num_letters=26_num_numbers=26_embedding_dim=512_num_heads=1_num_layers=2_num_training_data=None_number_symbolic_rep=True_seed=6/final_model.pt")
    experiment.compare_attention_heads(["experiments/exp_num_letters=26_num_numbers=26_embedding_dim=512_num_heads=1_num_layers=2_num_training_data=None_number_symbolic_rep=True_seed=2/final_model.pt",
                                        "experiments/exp_num_letters=26_num_numbers=26_embedding_dim=512_num_heads=1_num_layers=2_num_training_data=None_number_symbolic_rep=True_seed=6/final_model.pt"])
    # experiment.pred_1_head_model("experiments/exp_num_letters=26_num_numbers=26_embedding_dim=512_num_heads=1_num_layers=2_num_training_data=None_number_symbolic_rep=True_seed=2/final_model.pt")
    # experiment.train(steps=args.num_steps, batch_size=32)