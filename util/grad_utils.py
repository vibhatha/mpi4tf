import numpy as np
from tensorflow.python.framework.ops import EagerTensor
from util import tensor_utils as tutils


def grad_reduce(grad: list = None):
    new_grad: list = []
    if grad is None:
        raise Exception("Grad is None")
    else:
        for index, _et in enumerate(grad):
            et: EagerTensor = _et
            et_npy: np.ndarray = et.numpy()
            val = et_npy * 2
            tensor = tutils.to_tensor(val)
            new_grad.append(tensor)
            #print("Index {}, Et Type {}, Tensor Type {}".format(index, type(et), type(tensor)))
            # print(index, et_npy.shape)
    grad.clear()
    return new_grad
