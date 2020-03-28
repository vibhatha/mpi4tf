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
forward_time = []
backward_time = []
loss_time = []
grad_update_time = []


def train_step(images, labels):
    with tf.GradientTape() as tape:
        t1 = time.time()
        logits = mnist_model(images, training=True)
        forward_time.append(time.time() - t1)
        # Add asserts to check the shape of the output.
        # tf.debugging.assert_equal(logits.shape, (mini_batch_size, 10))
        t1 = time.time()
        loss_value = loss_object(labels, logits)
        loss_time.append(time.time() - t1)

    loss_history.append(loss_value.numpy().mean())
    t1 = time.time()
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    backward_time.append(time.time() - t1)
    t1 = time.time()
    grads = gutils.grad_reduce(grad=grads)
    t2 = time.time()
    total_allreduce_time.append(t2 - t1)
    t1 = time.time()
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
    grad_update_time.append(time.time() - t1)


train_step_calls: int = 0


def train(epochs) -> int:
    local_train_steps: int = 0
    for epoch in range(epochs):
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(images, labels)
            local_train_steps += 1
        print('Epoch {} finished, Batch {}'.format(epoch, batch))
    return local_train_steps


import time


epochs = 1
t1 = time.time()
train_step_calls = train(epochs=epochs)

training_time = (time.time() - t1) / epochs

train_step_calls = int(train_step_calls / epochs)

sample_size = int(mini_batch_size * train_step_calls)
communication_time = sum(total_allreduce_time) / epochs
forward_time_s = sum(forward_time) / epochs
backward_time_s = sum(backward_time) / epochs
loss_time_s = sum(loss_time) / epochs
grad_update_time_s = sum(grad_update_time) / epochs

print("CPU TIME : {}, Communication Time {}".format(training_time, communication_time))

world_rank = dist.get_world_rank(comm)
if world_rank == 0:
    with open(stats_file, "+a") as fp:
        fp.write(
            "{:d},{:d},{:d},{:d},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(dist.get_world_size(comm),
                                                                                            sample_size,
                                                                                            mini_batch_size,
                                                                                            train_step_calls,
                                                                                            training_time,
                                                                                            training_time - communication_time,
                                                                                            communication_time,
                                                                                            forward_time_s,
                                                                                            backward_time_s,
                                                                                            loss_time_s,
                                                                                            grad_update_time_s))

dist.finalize()
