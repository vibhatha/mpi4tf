import mpi4tf.core.distributed as dist
import tensorflow as tf

tf.executing_eagerly()

dist.initialize()

comm = dist.get_comm()

world_rank:int = dist.get_world_rank()
world_size:int = dist.get_world_size()

print("World Rank {}, World Size {}".format(world_rank, world_size))

a = tf.constant([1.0, 2.0, 3.0])

a_local = a * world_rank

a_global = dist.all_reduce(a_local, dist.ReduceOp.SUM)

print("a_local {}, a_global {}".format(a_local, a_global))

dist.finalize()