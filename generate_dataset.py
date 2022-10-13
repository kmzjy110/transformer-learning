import numpy as np
import json
NUMBERs = list(range(1,27))
LETTERs = ['A', 'B', 'C','D', 'E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
eps = 0.2
def generate_in_context_dataset(dataset_len = 5000, num_numbers=26, num_letters=26, save_path=None, seed=None):
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
def generate_maximal_char_dataset(dataset_len = 5000, num_numbers=26, num_letters=26, save_path=None):
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



