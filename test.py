dir1="experiments/exp_num_letters=26_num_numbers=26_embedding_dim=512_num_heads=1_num_layers=2_num_training_data=None_number_symbolic_rep=True_model_seed=12_data_seed=11_lr=0.0001_batch_size=32_max_steps=2000000_num_warmup_steps=5000_test/start_0_stop_10000_seed_11"
dir2="experiments/exp_num_letters=26_num_numbers=26_embedding_dim=512_num_heads=1_num_layers=2_num_training_data=None_number_symbolic_rep=True_model_seed=12_data_seed=11_lr=0.0001_batch_size=32_max_steps=2000000_num_warmup_steps=5000_test/start_500_stop_1500_seed_11"
import torch
import os


def get_model_diff(model1_state_dict, model2_state_dict):

    with torch.no_grad():
        l2_norm = 0

        for key in model1_state_dict.keys():

            l2_norm += torch.linalg.norm(model1_state_dict[key] - model2_state_dict[key]) ** 2

        return l2_norm.detach().cpu().numpy()

def get_optimizer_diff(model1_dict, model2_dict):
        current_diff = 0
        for key in model1_dict.keys():
            current_diff+=get_model_diff(model1_dict[key], model2_dict[key])

        return current_diff

for i in [500, 1000, 1500, 10000]:
    model1 = torch.load(os.path.join(dir1, f"training_step_{i}.pt"),map_location=torch.device('cpu'))
    model2 = torch.load(os.path.join(dir2, f'training_step_{i}.pt'),map_location=torch.device('cpu'))
    print(f"step {i}")

    print("diff of optimizer state dict")
    print(get_optimizer_diff(model1['optimizer_state_dict']['state'], model2['optimizer_state_dict']['state']))

    print('diff of lr scheduler state dict')
    print(model1['lr_scheduler_state_dict'])
    print(model2['lr_scheduler_state_dict'])

    print('diff of model state dict')
    print(get_model_diff(model1['model_state_dict'], model2['model_state_dict']))

    print("training seed:")
    print(model1['training_seed'])
    print(model2['training_seed'])

    # print("loss")
    # print(model1['loss'])
    # print(model2['loss'])
