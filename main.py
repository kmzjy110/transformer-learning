import generate_dataset
import transformer_model
import train
# data = generate_dataset.generate_dataset(5000)
train.model_train(50,32,5e-4,'model')