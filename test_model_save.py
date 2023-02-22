from experiment import Experiment
import torch
import os
def test_model_save():
    max_steps=2500
    interval=500
    experiment = Experiment(embedding_dim=512, num_heads=1,
                            num_layers=2, num_numbers=26, num_letters=26,
                            num_training_data=None, number_symbolic_rep=True,
                            data_seed=1, model_seed=2,
                            all_experiment_folder='experiments', max_steps=max_steps,
                            produce_plots=False, save_model_every_num_steps=interval
                            )
    experiment.train(batch_size=32)
    for i in range(interval,max_steps,interval):
        new_exp  = Experiment(embedding_dim=512, num_heads=1,
                            num_layers=2, num_numbers=26, num_letters=26,
                            num_training_data=None, number_symbolic_rep=True,
                            data_seed=1, model_seed=2,
                            all_experiment_folder='experiments', max_steps=max_steps,
                            produce_plots=False, save_model_every_num_steps=interval
                            )
        new_exp.train(batch_size=32,start_idx=i, model_save_dir=f'test_{i}/', model_load_dir=None)
        for j in range(i, max_steps+interval, interval):
            original_model_states = torch.load(os.path.join(experiment.experiment_dir, 'training_step_' + str(j) + '.pt'))
            new_model_states = torch.load(os.path.join(experiment.experiment_dir, f'test_{i}/','training_step_' + str(j) + '.pt'))
            experiment.model.load_state_dict(original_model_states['model_state_dict'])
            new_exp.model.load_state_dict(new_model_states['model_state_dict'])
            diff = get_model_diff(experiment.model, new_exp.model)

            print(f"model trained to step {j} from step: {i}, diff with direct trained model at step {j}: {diff}")





def get_model_diff(model1, model2):
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()
    with torch.no_grad():
        l2_norm = 0

        for key in model1_state_dict.keys():
            l2_norm += torch.linalg.norm(model1_state_dict[key] - model2_state_dict[key]) ** 2

        return l2_norm.detach().cpu().numpy()

test_model_save()