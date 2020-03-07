import tensorflow as tf

tf.executing_eagerly()
tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


with tf.device('/GPU:0'):
    c = tf.matmul(a,b)


print(c)
print(c.numpy())

print("GPU 0 Initi W")
with tf.device('/GPU:0'):
    w = tf.Variable([[1.0]])

print("GPU 0 Calculate loss")
with tf.device('/GPU:0'):
    with tf.GradientTape() as tape:
        loss = w * w

print("Loss : {}, {}".format(loss, loss.numpy()))

print("GPU 1 Calculate Gradient")
with tf.device('/GPU:1'):
    grad = tape.gradient(loss, w)

print("Grad : {}, {}".format(grad, grad.numpy()))  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)

print(" Old W : {} ".format(w))

print("GPU 2 Update Weights")

with tf.device('/GPU:2'):
    w = w - 0.01 * grad

print(" New W : {} ".format(w))

