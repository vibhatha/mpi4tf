# -*- coding: utf-8 -*-
"""tensorflow_mnist_gradient_tape.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vtXQnFKliVldGm6uj3qhZGn5ize_Ecy9
"""

# !pip install tensorflow-gpu==2.0.0b1

# Commented out IPython magic to ensure Python compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals
import os

from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf

import logging

logging.getLogger('tensorflow').setLevel(logging.FATAL)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import core.distributed as dist
from util import data_utils as dutils
from util import grad_utils as gutils

stats_file = "tf_stats_data_parallel_mnist.csv"

dist.initialize()

comm = dist.get_comm()

tf.executing_eagerly()

(mnist_train_images, mnist_train_labels), ((mnist_test_images, mnist_test_labels)) = tf.keras.datasets.mnist.load_data()

world_rank = dist.get_world_rank()
world_size = dist.get_world_size()

sequential_mini_batch_size = 600
mini_batch_size = int(sequential_mini_batch_size / world_size)
# picking constant mini-batch size
mini_batch_size = 25

print(mnist_train_images.shape, mnist_train_labels.shape, mnist_test_images.shape, mnist_test_labels.shape)

mnist_train_images_local = dutils.get_data_partition(mnist_train_images, world_size, world_rank)
mnist_train_labels_local = dutils.get_data_partition(mnist_train_labels, world_size, world_rank)

mnist_test_images_local = dutils.get_data_partition(mnist_test_images, world_size, world_rank)
mnist_test_labels_local = dutils.get_data_partition(mnist_test_labels, world_size, world_rank)

print(mnist_train_images_local.shape, mnist_train_labels_local.shape, mnist_test_images_local.shape,
      mnist_test_labels_local.shape)

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_train_images_local[..., tf.newaxis] / 255, tf.float32),
     tf.cast(mnist_train_labels_local, tf.int64)))
dataset = dataset.shuffle(1000).batch(mini_batch_size)

# Build the model
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu',
                           input_shape=(None, None, 1)),
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

for images, labels in dataset.take(1):
    print("Logits: ", mnist_model(images[0:1]).numpy())

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []

dist_loss_history = []

import time

total_allreduce_time = []


def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)

        # Add asserts to check the shape of the output.
        tf.debugging.assert_equal(logits.shape, (mini_batch_size, 10))

        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    t1 = time.time()
    grads = gutils.grad_reduce(grad=grads)
    t2 = time.time()
    total_allreduce_time.append(t2 - t1)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))


def train(epochs):
    for epoch in range(epochs):
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(images, labels)
        print('Epoch {} finished, Batch {}'.format(epoch, batch))


import time

t1 = time.time()
train(epochs=1)

training_time = time.time() - t1
communication_time = sum(total_allreduce_time) / len(total_allreduce_time)
print("CPU TIME : {}, Communication Time {}".format(training_time, communication_time))

world_rank = dist.get_world_rank(comm)
if world_rank == 0:
    with open(stats_file, "+a") as fp:
        fp.write("{:d},{:.6f},{:.6f},{:.6f}\n".format(dist.get_world_size(comm), training_time,
                                                      training_time - communication_time, communication_time))

dist.finalize()
