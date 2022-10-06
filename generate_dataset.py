import numpy as np
import json
numbers = list(range(1,27))
letters = ['A', 'B', 'C','D', 'E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
def generate_dataset(dataset_len = 5000, distribution_threshold=0.5):
    data = []
    num_integer_datapoints = 0
    num_decimal_datapoints = 0
    for i in range(dataset_len):
        #sample 1 to 26 integers
        #sample a number closest to one of these (one and only one...?); each number has two adjacent numbers that are closest to them (+- 0.1)
        current_len = np.random.randint(1,26)
        generated_numbers = np.random.choice(numbers, current_len, replace=False).tolist()
        generated_letters = np.random.choice(letters, current_len, replace=False).tolist()
        sol_num = np.random.choice(generated_numbers)
        sol_letter = generated_letters[generated_numbers.index(sol_num)]
        generated_numbers_sorted = sorted(generated_numbers)
        higher_closest_integers = []
        lower_closest_integers = []
        sol_num_index_in_sorted = generated_numbers_sorted.index(sol_num)
        prompt = None
        def get_higher_midpoint(lower_num, higher_num):
            if (lower_num+higher_num)%2 ==0:
                return int(((lower_num+higher_num)/2)+1)
            else:
                return int(np.ceil((float(lower_num) + float(higher_num))/2))
        if sol_num_index_in_sorted!=0:
            if sol_num - generated_numbers_sorted[sol_num_index_in_sorted-1] >2:
                r = range(get_higher_midpoint(generated_numbers_sorted[sol_num_index_in_sorted-1], sol_num), sol_num) # not including sol_num in the lower range
                lower_closest_integers.extend(r)

        if sol_num_index_in_sorted!=len(generated_numbers_sorted)-1:
            if generated_numbers_sorted[sol_num_index_in_sorted+1] - sol_num >2:
                r= range(sol_num, get_higher_midpoint(sol_num, generated_numbers_sorted[sol_num_index_in_sorted +1]))
                higher_closest_integers.extend(r)
        if sol_num_index_in_sorted==0 and sol_num!=1:
            lower_closest_integers.extend(range(1, sol_num))

        if sol_num_index_in_sorted == len(generated_numbers_sorted)-1 and sol_num!=26:
            higher_closest_integers.extend(range(sol_num, 27))
        if np.random.uniform()<=distribution_threshold:
            num_decimal_datapoints +=1
            if np.random.uniform() <0.5:
                prompt = sol_num-0.1
            else:
                prompt = sol_num+0.1
        else:
            num_integer_datapoints+=1
            if len(higher_closest_integers)==0 and len(lower_closest_integers)==0:
                prompt = sol_num
            else:
                random_num = np.random.uniform()
                if not lower_closest_integers:
                    prompt = np.random.choice(higher_closest_integers)
                elif not higher_closest_integers:
                    prompt = np.random.choice(lower_closest_integers)
                elif random_num<0.5:
                    prompt = np.random.choice(higher_closest_integers)
                else:
                    prompt = np.random.choice(lower_closest_integers)
        sequence = ""
        for j in range(current_len):
            sequence = sequence + str(generated_numbers[j]) + " " + generated_letters[j] + ' '
        data.append({"sequence": sequence, "prompt": str(prompt), "sol": sol_letter})
    print(num_integer_datapoints)
    print(num_decimal_datapoints)
    with open('data.json', 'w+') as out_file:
        json.dump(data,out_file)
    return data

