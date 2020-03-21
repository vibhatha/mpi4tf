import numpy as np
from tensorflow.python.framework.ops import EagerTensor
import core.distributed as dist


def grad_reduce(grad: list = None):
    new_grad: list = []
    if grad is None:
        raise Exception("Grad is None")
    else:
        for index, _et in enumerate(grad):
            et: EagerTensor = _et
            et_npy: np.ndarray = et.numpy()
            val = et_npy * 2
            tensor = dist.all_reduce(tensor=et, op=dist.ReduceOp.SUM)
            new_grad.append(tensor)
    grad.clear()
    return new_grad
