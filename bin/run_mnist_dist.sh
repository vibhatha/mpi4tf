par=$1
mpirun --mca coll_base_verbose 1 --mca btl_openib_allow_ib 1 -n ${par} python3 examples/data_parallel/data_parallel_v1.py
