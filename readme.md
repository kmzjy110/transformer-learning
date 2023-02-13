Running the code:

```/hzhao/transformer-learning/gpu python experiment.py```

Initializing the experiment in experiment.py:

```
experiment = Experiment(embedding_dim=512, num_heads=1,
                       num_layers=2, num_numbers=26, num_letters=26,
                       num_training_data=args.num_training_data, number_symbolic_rep=True, seed=2, save_model_every_num_steps=5000)

#init arguments:
#lr: calculated with get_linear_schedule_with_warmup
#produce_plots: save attention heatmap during training
#save_attn_every_num_steps: how many steps in between to save attention heatmap
#plot_attn_seq_len: What sequence length to use to plot the attention map

```

Train the model:
```
experiment.train(steps=args.num_steps, batch_size=32,
            final_model_state_dict=torch.load(f'experiments/exp_num_letters=26_num_numbers=26_embedding_dim=512_num_heads=1_num_layers=2_num_training_data=None_number_symbolic_rep=True_seed=6/model_step_955000.pt'))

#function arguments:
#final_model_state_dict: if passed in, the train function will evaluate the average model between the model at step i and the final model

```

Various functions:
Generate attention plots given a model path


```
experiment = Experiment(embedding_dim=512, num_heads=1,
                       num_layers=2, num_numbers=26, num_letters=26,
                       num_training_data=args.num_training_data, number_symbolic_rep=True, seed=2, save_model_every_num_steps=5000)


experiment.generate_model_attention_plots("experiments/exp_num_letters=26_num_numbers=26_embedding_dim=512_num_heads=1_num_layers=2_num_training_data=None_number_symbolic_rep=True_seed=6/final_model.pt")
```

Hard code predict attention pattern
```
experiment = Experiment(embedding_dim=512, num_heads=1,
                       num_layers=2, num_numbers=26, num_letters=26,
                       num_training_data=args.num_training_data, number_symbolic_rep=True, seed=2, save_model_every_num_steps=5000)


 experiment.pred_1_head_model("experiments/exp_num_letters=26_num_numbers=26_embedding_dim=512_num_heads=1_num_layers=2_num_training_data=None_number_symbolic_rep=True_seed=2/final_model.pt")
```

