
"""
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ._core import BaseModel
from .preprocessor import prep_func
from .postprocessor import post_func
from . import config

import os
import numpy as np
import tensorflow as tf


__all__ = ['TensorflowModel']


def input_dataset_fn(features, labels, shuffle, num_epochs, batch_size):
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


class TensorflowModel(BaseModel):

    def __init__(self, input_dim=None, learning_rate=None, dirpath=None):

        self.INPUT_DIM = None

        if dirpath:
            self.load(dirpath)
        elif input_dim:
            if learning_rate is None:
                raise Exception(
                    "When 'input_dim' is given, 'learning_rate' must be given too."
                )
            self.build(input_dim, learning_rate)
        else:
            raise Exception("'input_dim' or 'dirpath' must be given.")

    def build(self, input_dim, learning_rate):
        self.INPUT_DIM = input_dim

        Dense = tf.keras.layers.Dense
        model = tf.keras.Sequential(
        [
            Dense(
                100, activation=tf.nn.relu,
                kernel_initializer='uniform',
                input_shape=input_dim,
            ),
            Dense(50, activation=tf.nn.relu),
            Dense(1, activation=tf.nn.sigmoid)
        ])

        optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)

        # Compile tf.keras model
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        self.model = model

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        # Case 1: `prep_func` is made for the batch `X`
        X = prep_func(X)
        # Case 2: `prep_func` is made for each input in `X`
        X = np.array([prep_func(x) for x in X])
        return X

    def postprocess(self, y_hat: np.ndarray) -> np.ndarray:
        # Case 1: `post_func` is made for the batch `y_hat`
        y_hat = post_func(y_hat)
        # Case 2: `post_func` is made for each output in `y_hat`
        y_hat = np.array([post_func(yh) for yh in y_hat])
        return y_hat

    def train(self, training_dataset,
              num_train_examples, num_epochs,
              validation_dataset, callbacks=None):

        self.model.fit(
            training_dataset,
            steps_per_epoch=None,
            # steps_per_epoch=int(num_train_examples / args.batch_size),
            epochs=num_epochs,
            validation_data=validation_dataset,
            validation_steps=1,
            verbose=1,
            callbacks=callbacks)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def predict(self, inputs):
        prep = self.preprocess(inputs)
        y_hat = self.model.predict(
            prep, batch_size=None, verbose=0, 
            steps=None, callbacks=None,
            max_queue_size=10, workers=1,
            use_multiprocessing=False,
        )
        return self.postprocess(y_hat)

    def save(self, dirpath,
             overwrite=True, include_optimizer=True,
             *args, **kwargs):
        self.model.save(dirpath,
                        overwrite=overwrite,
                        include_optimizer=include_optimizer,
                        save_format='tf',
                        *args, **kwargs)

    def load(self, dirpath, *args, **kwargs):
        self.model = tf.keras.models.load_model(
            dirpath, *args, **kwargs
        )
        self.INPUT_DIM = self.model.input_shape[1:]
