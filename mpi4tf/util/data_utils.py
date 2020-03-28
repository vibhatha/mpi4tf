import numpy as np

def get_data_partition(dataset:np.ndarray, partitions:int, partition_id: int):
    samples: int = len(dataset)
    samples_per_local: int = int(samples / partitions)
    start: int = samples_per_local * partition_id
    end: int = start + samples_per_local
    return dataset[start:end]
