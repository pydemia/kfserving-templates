import os
import numpy as np
from .model import SKLearnModel


# Load train & valid data
train_record_num = 200
train_x = np.random.rand(train_record_num, 10)
train_y = np.random.randint(2, size=(train_record_num, 1))

print(f'train_x.shape: {train_x.shape}')
print(f'train_y.shape: {train_y.shape}')

eval_record_num = 50
eval_x = np.random.rand(eval_record_num, 10)
eval_y = np.random.randint(2, size=(eval_record_num, 1))

print(f'eval_x.shape: {eval_x.shape}')
print(f'eval_y.shape: {eval_y.shape}')


# Create a Model
INPUT_DIM = train_x.shape[1:]  # (10, )
model1 = SKLearnModel()


# Create Datasets
NUM_EPOCHS = 10
BATCH_SIZE = 16

# Training
model1.train(train_x, train_y)


# Evaluation
eval_res1 = model1.evaluate(eval_x, eval_y)

# Save
model1.save(f'{os.path.dirname(__file__)}/../../models', compress=1)


# Load & Reuse
model2 = SKLearnModel(dirpath=f'{os.path.dirname(__file__)}/../../models')

eval_res2 = model2.evaluate(eval_x, eval_y)

print(f'model1.evaluate: {eval_res1}')
print(f'model2.evaluate: {eval_res2}')

# model2.train(train_x, train_y)

new_x = np.random.rand(1, 10)
print(new_x)

pred_res1 = model1.predict(new_x)
pred_res2 = model2.predict(new_x)

print(f'model1.predict: {pred_res1}')
print(f'model2.predict: {pred_res2}')
