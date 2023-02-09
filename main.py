import pickle
import numpy
import matplotlib.pyplot as plt
data= []
seeds = [2]
# for i in seeds:
#
#     with open('26_1_seed_'+str(i)+".pkl", 'rb') as f:
#         data.append(pickle.load(f))
#
# for index, data_seed in enumerate(data):
#     x = [item['step'] for item in data_seed]
#     y = [item['acc'] for item in data_seed]
#     plt.plot(x,y, label ="seed "+ str(seeds[index]) +" model acc")
#
# for index, data_seed in enumerate(data):
#     x = [item['step'] for item in data_seed]
#     y = [item['hardcode_pred_acc'] for item in data_seed]
#     plt.plot(x,y, label ="seed "+ str(seeds[index]) +" hard code prediction accuracy")

lc = pickle.load(open('2_lc.pkl','rb'))
# x = [item['step'] for item in lc]
# y = [item['acc'] for item in lc]
# plt.plot(x,y, label ="averaged model accuracy (between final model and step x model)")
#
# x = [item['step'] for item in lc]
# y = [item['hardcode_pred_acc'] for item in lc]
# plt.plot(x,y, label ="hardcode pred accuracy for averaged model")

x = [item['step'] for item in lc]
y = [item['l2_norm'] for item in lc]
plt.plot(x,y, label ="l2 norm of model")
plt.legend()
plt.show()

