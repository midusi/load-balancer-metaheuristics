from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import numpy as np
import time
import resource
from multiprocessing import Process, Queue
import timeit


def process_data(q: Queue, rdd):
    # Read in pySpark DataFrame partition
    data = list(rdd)

    # Generate random data using Numpy
    rand_data = np.random.random(int(1e7))

    # Apply the `int` function to each element of `rand_data`
    for i in range(len(rand_data)):
        e = rand_data[i]
        int(e)

    # Return a single `0` value
    q.put([[0]])


def toy_example_with_process(rdd):
    # `used_memory` size should not be increased on every call to toy_example as
    # the previous call memory should be released
    used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    q = Queue()
    p = Process(target=process_data, args=(q, rdd))
    p.start()
    _process_result = q.get()
    p.join()

    return [[used_memory]]


def toy_example(rdd):
    # `used_memory` size should not be increased on every call to toy_example as
    # the previous call memory should be released
    used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Read in pySpark DataFrame partition
    data = list(rdd)

    # Generate random data using Numpy
    rand_data = np.random.random(int(1e7))

    # Apply the `int` function to each element of `rand_data`
    for i in range(len(rand_data)):
        e = rand_data[i]
        int(e)

    return [[used_memory]]


def worker_reuse_false(df):
    """Allocations are in the mapPartitions function but the `spark.python.worker.reuse` is set to 'false'
    and prevents memory leaks"""
    memory_usage = df.rdd.mapPartitions(toy_example).collect()
    print(memory_usage)  # Just for debugging, remove


def with_process(df):
    """Allocations are inside a new Process. Memory is released by the OS"""
    memory_usage = df.rdd.mapPartitions(toy_example_with_process).collect()
    print(memory_usage)  # Just for debugging, remove


iterations = 10

# Timeit with `spark.python.worker.reuse` = 'false'
conf = SparkConf().setMaster(MASTER_CONNECTION_URL).setAppName(f"Memory leak reuse false {time.time()}")
conf = conf.set("spark.python.worker.reuse", 'false')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
df = sqlContext.range(0, int(1e5), numPartitions=16)
worker_reuse_time = timeit.timeit(lambda: worker_reuse_false(df), number=iterations)
print(f'Worker reuse: {round(worker_reuse_time, 3)} seconds')


# Timeit with external Process
sc.stop()  # Needed to set a new SparkContext config
conf = SparkConf().setMaster(MASTER_CONNECTION_URL).setAppName(f"Memory leak with process {time.time()}")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
df = sqlContext.range(0, int(1e5), numPartitions=16)
with_process_time = timeit.timeit(lambda: with_process(df), number=iterations)
print(f'With process: {round(with_process_time, 3)} seconds')
