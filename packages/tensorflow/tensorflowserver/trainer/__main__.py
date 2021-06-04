import os
import numpy as np
from .model import input_dataset_fn, TensorflowModel


# Load train & valid data
train_num = 200
train_x = np.random.rand(train_num, 10)
train_y = np.random.randint(2, size=(train_num, 1))

print(train_x.shape, train_y.shape)

eval_num = 50
eval_x = np.random.rand(eval_num, 10)
eval_y = np.random.randint(2, size=(eval_num, 1))

print(eval_x.shape, eval_y.shape)


# Create a Model
INPUT_DIM = train_x.shape[1:]  # (10, )
model = TensorflowModel(input_dim=INPUT_DIM, learning_rate=0.1)


# Create Datasets
NUM_EPOCHS = 10
BATCH_SIZE = 16
training_dataset = input_dataset_fn(
    features=train_x,
    labels=train_y,
    shuffle=True,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
)

validation_dataset = input_dataset_fn(
    features=eval_x,
    labels=eval_y,
    shuffle=False,
    num_epochs=NUM_EPOCHS,
    batch_size=eval_num,
)


# Training
model.train(training_dataset,
            train_num,
            NUM_EPOCHS,
            validation_dataset)


# Evaluation
model.evaluate(validation_dataset)


# Save
model.save(f'{os.path.dirname(__file__)}/../../models/1', overwrite=True)


# Load & Reuse
model2 = TensorflowModel(dirpath=f'{os.path.dirname(__file__)}/../../models/1')

model2.INPUT_DIM
model2.model.summary()

model2.train(training_dataset,
             train_num,
             10,
             validation_dataset)

model2.predict(np.random.rand(1, 10))
