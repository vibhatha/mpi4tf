# mpi4tf

This is a basic distributed training enabled platform for training Tensorflow. 

## Pre-requisites

Install OpenMPI 4.x.x or any other MPI implementation. 

## Install via Pip

```bash
pip3 install mpi4tf
```

## Install From Source

Clone the `mpi4tf` repo then, 

```bash
python3 setup.py install
```

## Development Mode

In the development mode use the following command to build the libraries. 

```bash
python3 setup.py develop
```

## Test

```bash
mpirun -n 4 python3 test/test_mpi.py
```

## MNIST Data Parallel Demo

Run with Parallelism 4

```bash
./bin/run_mnist_dist.sh 4
```

## Notes

As this is a MPI backend you can use all the MPI flags to add different functionality
in running experiments. 