import mpi4py
from tensorflow.python.framework.ops import EagerTensor
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI
from enum import Enum
import numpy as np
from mpi4tf.util import tensor_utils as tutils


class ReduceOp(Enum):
    SUM = MPI.SUM
    MAX = MPI.MAX
    MIN = MPI.MIN
    PROD = MPI.PROD


def initialize():
    if not MPI.Is_initialized():
        MPI.Init()
    global _comm
    _comm = MPI.COMM_WORLD


def finalize():
    if not MPI.Is_finalized():
        MPI.Finalize()


def _get_comm():
    return MPI.COMM_WORLD


def get_comm():
    return _get_comm()


def get_world_rank(comm: MPI.COMM_WORLD = None) -> int:
    if comm is None:
        comm = _get_comm()
    return comm.Get_rank()


def get_world_size(comm: MPI.COMM_WORLD = None) -> int:
    if comm is None:
        comm = _get_comm()
    return comm.Get_size()


def all_reduce(tensor: EagerTensor, op=ReduceOp.SUM, comm: MPI.COMM_WORLD = None) -> EagerTensor:
    param_numpy = tensor.numpy()
    original_shape = param_numpy.shape
    param_numpy_flatten = param_numpy.flatten()
    param_output = np.empty(param_numpy_flatten.shape, dtype=param_numpy.dtype)
    if comm is None:
        comm = _get_comm()
    comm.Allreduce(param_numpy_flatten, param_output, op=op.value)
    param_output = np.reshape(param_output, original_shape)
    tensor = tutils.to_tensor(param_output)
    return tensor