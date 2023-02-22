import pickle
import numpy
import matplotlib.pyplot as plt
data= []
seeds = [6]
for i in seeds:

    with open('26_2_'+str(i)+".pkl", 'rb') as f:
        data.append(pickle.load(f))
print(data)
for index, data_seed in enumerate(data):
    x = [item['step'] for item in data_seed]
    y = [item['acc'] for item in data_seed]
    plt.plot(x,y, label ="seed "+ str(seeds[index]) +" model acc")

for index, data_seed in enumerate(data):
    x = [item['step'] for item in data_seed]
    y = [item['hardcode_pred_acc'] for item in data_seed]
    plt.plot(x,y, label ="seed "+ str(seeds[index]) +" hard code prediction accuracy")

lc = pickle.load(open('26_2_6.pkl','rb'))


max_l2 = max([item['l2_norm'] for item in lc])
min_l2 = min([item['l2_norm'] for item in lc])
x = [item['step'] for item in lc]
y = [(item['l2_norm'] - min_l2)/(max_l2-min_l2) for item in lc]
plt.plot(x,y, label ="l2 norm of model (normalized)")
plt.legend()
plt.show()

print(lc)