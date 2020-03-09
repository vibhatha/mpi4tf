import tensorflow as tf
import numpy as np
import time

numpy_to_tensor_times = []
tensor_to_numpy_times = []

arr_size = 1000000
repititions = 1000

for i in range(repititions):
    array = np.random.rand(arr_size, 1)
    t1 = time.time()
    tensor = tf.convert_to_tensor(array, dtype=tf.float32)
    t2 = time.time()
    numpy_to_tensor_times.append(t2 - t1)
    t1 = time.time()
    numpy_val = tensor.numpy()
    t2 = time.time()
    tensor_to_numpy_times.append(t2-t1)

print("Nump to Tensor Average Time : {}".format(sum(numpy_to_tensor_times) / len(numpy_to_tensor_times)))
print("Tensor to Numpy Average Time : {}".format(sum(tensor_to_numpy_times) / len(tensor_to_numpy_times)))