import pickle
import numpy
import matplotlib.pyplot as plt
data= []
seeds = range(9)
for i in seeds:

    with open('26_1_seed_'+str(i)+".pkl", 'rb') as f:
        data.append(pickle.load(f))

for index, data_seed in enumerate(data):
    x = [item['step'] for item in data_seed]
    y = [item['acc'] for item in data_seed]
    plt.plot(x,y, label ="seed "+ str(seeds[index]) )
plt.title("1 head convergence")
plt.legend()
plt.show()

