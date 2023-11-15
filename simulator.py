import time
from copy import deepcopy
from math import ceil
from typing import Literal, Final, Optional, Union
import binpacking
import numpy as np
from mknapsack import solve_multiple_knapsack
from metaheuristics import get_random_subset_of_features
import matplotlib.pyplot as plt

# Dict with the partition identifier as key and the worker identifier on which the partition will be computed as value
PartitionToWorker = dict[int, str]

# Dict with the worker identifier as key and some delay to apply in every execution. For example, 'worker_0': 1.75
# will generate a delay of 75% in the worker_0 execution time for every RDD.
DelayForWorker = dict[str, float | None]

# Dict with the worker identifier as key and the list of execution times as value
WorkerTimes = dict[str, list[float]]

WorkerIdleTime = dict[str, float]

# Idle/Execution times structure for bar charts. Keys are workers, values is a List of lists with the iteration number
# and the time.
WorkerBarTimes = dict[str, list[list[float]]]

# Types of strategy to define partitions
PartitionStrategy = Literal['n_stars', 'binpacking', 'smart']

# Strategy to define partitions
# STRATEGY: Final[PartitionStrategy] = 'n_stars'
# STRATEGY: Final[PartitionStrategy] = 'binpacking'
STRATEGY: Final[PartitionStrategy] = 'smart'

# Seed for reproducibility
RANDOM_SEED: Final[Optional[int]] = 12

# If true, adds a delay to worker number 2
ADD_DELAY_TO_WORKER_2: Final[bool] = False

# Some constants
N_WORKERS: Final = 3
N_STARS: Final = 6
N_FEATURES: Final = 5
ITERATIONS: Final = 10
# ITERATIONS: Final = 1

# To print useful information
DEBUG: Final = True


def __print_all_knapsack(weights, capacities):
    print(f'Weights: {weights}')
    print(f'Weights sum: {sum(weights)}')
    print(f'Capacities: {capacities}')


class Rdd:
    """
    Simulates the RDD class of Spark. An RDD has a subset of features and a partition to define in which worker
    it will be computed.
    """
    partition: int
    subset: np.ndarray

    def __init__(self, partition: int, subset: np.ndarray):
        self.partition = partition
        self.subset = subset

    def __repr__(self):
        return f"Rdd(partition={self.partition}, n_features={np.count_nonzero(self.subset)})"


def collect(rdds: list[Rdd], partition_to_worker: PartitionToWorker,
            delay_for_worker: Optional[DelayForWorker]) -> tuple[WorkerTimes, WorkerIdleTime, DelayForWorker]:
    """
    Simulates the collect() method of Spark.
    :param rdds: List of RDD to evaluate.
    :param partition_to_worker: Dict with the partition identifier as key and the worker identifier on which the
    partition will be computed as value.
    :param delay_for_worker: Dict with the worker identifier as key and some delay to apply in every execution.
    """
    worker_execution_times: WorkerTimes = {}

    for rdd in rdds:
        worker_id = partition_to_worker[rdd.partition]

        worker_start = time.time()
        _current_fitness = __fitness_function(rdd.subset)
        worker_end = time.time()

        worker_execution_time = worker_end - worker_start

        if delay_for_worker is not None and worker_id in delay_for_worker:
            if DEBUG:
                print(f'Worker {worker_id} has a delay of {delay_for_worker[worker_id]}')
                print(f'Original execution time: {worker_execution_time} | New execution time: '
                      f'{worker_execution_time * delay_for_worker[worker_id]}')
            worker_execution_time += worker_execution_time * delay_for_worker[worker_id]

        # Stores the execution time of the worker
        if worker_id in worker_execution_times:
            worker_execution_times[worker_id].append(worker_execution_time)
        else:
            worker_execution_times[worker_id] = [worker_execution_time]

    # Gets max sum of execution times
    max_worker_time = -1
    for worker_id in worker_execution_times:
        sum_worker_time = np.sum(worker_execution_times[worker_id])

        if DEBUG:
            print(f'Worker {worker_id} has executed {len(worker_execution_times[worker_id])} RDDs in '
                  f'{sum_worker_time} seconds')

        if sum_worker_time > max_worker_time:
            max_worker_time = sum_worker_time

    # Stores the idle time for every worker. This is the difference between the max worker time and the sum
    # of the execution times of the worker
    idle_times: WorkerIdleTime = {}
    max_idle_time = -1
    for worker_id in worker_execution_times:
        idle_time = max_worker_time - np.sum(worker_execution_times[worker_id])
        idle_times[worker_id] = idle_time

        if idle_time > max_idle_time:
            max_idle_time = idle_time

    # Adds the percentage of delay for every worker. The worker with the highest idle time will have a delay of 0%,
    # and the rest of workers will have a delay proportional to their idle time.
    # TODO: fix this
    delay_percentage: DelayForWorker = {}
    for worker_id in worker_execution_times:
        delay_percentage[worker_id] = (max_idle_time - idle_times[worker_id]) / max_idle_time

    return worker_execution_times, idle_times, delay_percentage


def __fitness_function(subset: np.ndarray) -> int:
    """Simulates a fitness function. It will return the number of features in the subset, and it'll last the same amount
    of time as the number of features in the subset."""
    time.sleep(np.count_nonzero(subset) / 9)
    return len(subset)


def __predict(subset: np.ndarray, random_seed: Optional[int]) -> float:
    """Simulates the predict function. It will return the number of features +- some random epsilon."""
    if random_seed is not None:
        np.random.seed(random_seed)

    return np.count_nonzero(subset) + np.random.uniform(-1, 1)


def __generate_stars_and_partitions_bins(bins: list) -> dict[int, int]:
    """
    Generates a dict with the idx of the star and the assigned partition
    :param bins: Bins generated by binpacking
    :return: Dict where keys are star index, values are the Spark partition
    """
    stars_and_partitions: dict[int, int] = {}
    for partition_id, aux_bin in enumerate(bins):
        for star_idx in aux_bin.keys():
            stars_and_partitions[star_idx] = partition_id
    return stars_and_partitions


def __binpacking_strategy(rdds: list[Rdd]) -> list[Rdd]:
    res_rdd = deepcopy(rdds)

    stars_and_times: dict[str, float] = {}
    for (idx, rdd) in enumerate(res_rdd):
        random_seed = RANDOM_SEED + idx if RANDOM_SEED is not None else None
        stars_and_times[idx] = __predict(rdd.subset, random_seed)
    bins = binpacking.to_constant_bin_number(stars_and_times, N_WORKERS)  # n_workers is the number of bins

    if DEBUG:
        print("Stars (keys) and their predicted execution times (values):")
        print(f"\nRepartition among {N_WORKERS} bins:")
        print(bins)

    # Generates a dict with the idx of the star and the assigned partition
    stars_and_partitions = __generate_stars_and_partitions_bins(bins)

    if DEBUG:
        print(stars_and_partitions)

    # Assigns the partition to each RDD
    for idx, rdd in enumerate(res_rdd):
        rdd.partition = stars_and_partitions[idx]

    return res_rdd


def __smart_strategy(rdds: list[Rdd], workers_delay: dict[str, float]) -> list[Rdd]:
    res_rdd = deepcopy(rdds)

    # Dict with the partition identifier as key and the list of RDDs as value. Starts from 1 as 0 means not assigned
    # partition
    worker_tasks_partitions: {str: list[int]} = {i + 1: [] for i in range(N_WORKERS)}
    worker_debug_weights: {str: int} = {__get_worker_id(i): 0 for i in range(N_WORKERS)}

    n_elements = len(rdds)

    # Keeps track of the original index of every RDD to update their partitions
    rdds_idxs = list(range(n_elements))

    # Generates an array of 1 profit for every RDD
    tasks = [1 for _ in range(n_elements)]

    # Gets the weights of every RDD
    weights = []
    for (idx, rdd) in enumerate(res_rdd):
        random_seed = RANDOM_SEED + idx if RANDOM_SEED is not None else None
        prediction = __predict(rdd.subset, random_seed)
        weights.append(ceil(prediction))

    while len(tasks) > 0:
        # ...and N_WORKERS knapsacks with the same capacity (the sum of all the weights) / N_WORKERS
        weight_per_worker_equal = ceil(sum(weights) / N_WORKERS)
        min_weight = min(weights)
        weight_per_worker = max(weight_per_worker_equal, min_weight)

        capacities = [weight_per_worker for _ in range(N_WORKERS)]

        if DEBUG:
            print('Before applying delay:')
            __print_all_knapsack(weights, capacities)

        if workers_delay is not None:
            extra_capacity_for_best_worker = 0  # Sum of the extra capacity for the best worker
            idx_worker_without_delay = None
            for idx in range(N_WORKERS):
                old_capacity = capacities[idx]
                worker_id = __get_worker_id(idx)
                if workers_delay[worker_id] != 0.0:
                    delay_for_worker = workers_delay[worker_id]
                    extra_capacity_for_best_worker += delay_for_worker

                    capacities[idx] -= capacities[idx] * delay_for_worker
                    if DEBUG:
                        print(f'Worker {idx}: {old_capacity} -> {capacities[idx]}. Decremented in a '
                              f'~{round(delay_for_worker * 100, 2)}%')
                else:
                    if DEBUG:
                        print(f'Worker {idx} has no delay')
                    idx_worker_without_delay = idx

                capacities[idx] = ceil(capacities[idx])

            # The fastest worker will have more capacity
            capacities[idx_worker_without_delay] += (capacities[idx_worker_without_delay] *
                                                     extra_capacity_for_best_worker)
            capacities[idx_worker_without_delay] = ceil(capacities[idx_worker_without_delay])

            if DEBUG:
                print('After applying delay:')
                __print_all_knapsack(weights, capacities)
        elif DEBUG:
            print('No delay to apply')

        # Assign items into the knapsacks while maximizing profits
        # NOTE: check_inputs=0 is used to avoid checking the inputs in the Fortran code and raise an error
        # when the number of task is less than the number of workers
        start = time.time()
        res_knp = solve_multiple_knapsack(tasks, weights, capacities, method='mthm', method_kwargs={'check_inputs': 0})
        if DEBUG:
            end = time.time()
            print(f'solve_multiple_knapsack() elapsed time: {end - start}')

        tasks_to_remove = set()
        for idx, assigned_partition_id in enumerate(res_knp):
            # Bin_number == 0 means that the item was not assigned to any knapsack
            if assigned_partition_id != 0:
                worker_tasks_partitions[assigned_partition_id].append(rdds_idxs[idx])  # Assigns the task to the worker

                # Adds weight to the worker
                if DEBUG:
                    worker_debug_weights[__get_worker_id(assigned_partition_id - 1)] += weights[idx]

                # Removes the task from the list of tasks and weights
                tasks_to_remove.add(idx)

        # Removes already asigned tasks
        rdds_idxs = [task_idx for idx, task_idx in enumerate(rdds_idxs) if idx not in tasks_to_remove]
        tasks = [task for idx, task in enumerate(tasks) if idx not in tasks_to_remove]
        weights = [weight for idx, weight in enumerate(weights) if idx not in tasks_to_remove]

        if len(tasks) > 0 and DEBUG:
            print(f'Pending {len(tasks)} tasks. Backlog: {tasks}')

    # Assigns the partition to each RDD
    if DEBUG:
        print('worker_tasks_partitions')
        print(worker_tasks_partitions)

        print('Weights per worker')
        print(worker_debug_weights)

    for worker_id, rdds_idxs in worker_tasks_partitions.items():
        if worker_id == 0:
            continue

        for rdd_idx in rdds_idxs:
            res_rdd[rdd_idx].partition = worker_id - 1

    if DEBUG:
        print(res_rdd)

    return res_rdd


def __generate_partition_to_work_default() -> PartitionToWorker:
    """Generates partition to worker dict where every partition is assigned to a worker."""
    return {partition_id: __get_worker_id(partition_id) for partition_id in range(N_WORKERS)}


def __assign_partitions(rdds: list[Rdd], strategy: PartitionStrategy,
                        workers_delay: Optional[DelayForWorker] = None) -> tuple[list[Rdd], PartitionToWorker]:
    """Assigns the partition to each RDD depending on the strategy. TODO: implement smart strategy"""
    if strategy == 'binpacking':
        return __binpacking_strategy(rdds), __generate_partition_to_work_default()
    if strategy == 'n_stars':
        # Separates the RDDs in N_STARS groups
        len_rdds = len(rdds)
        return ([Rdd(i * N_WORKERS // len(rdds), rdds[i].subset) for i in range(len_rdds)],
                __generate_partition_to_work_default())
    if strategy == 'smart':
        return __smart_strategy(rdds, workers_delay), __generate_partition_to_work_default()


def __add_to_worker_dict(current_execution_times: Union[WorkerTimes, WorkerIdleTime],
                         time_worker: WorkerBarTimes, iteration: int):
    """Adds the execution/idle times of the current iteration to the dict with the execution/idle times per worker."""
    for worker_id in current_execution_times:
        if worker_id in time_worker:
            time_worker[worker_id].append([iteration, np.sum(current_execution_times[worker_id])])
        else:
            time_worker[worker_id] = [[iteration, np.sum(current_execution_times[worker_id])]]


def generate_bar_charts(data: WorkerBarTimes, title: str, data_type: Literal['Execution', 'Idle']):
    """
    Plots a bar chart
    :param data: Dictionary with the worker name and the number of id
    :param title: Title to show in the bar chart
    :param data_type: To show 'Idle' or 'Execution' in bar chart title.
    """
    _fig, ax = plt.subplots()

    # Adds some text for labels, title and axes ticks
    ax.set_ylabel(f'{data_type} time (seconds)')
    fig_title = f'{data_type} time per worker. {title}'
    ax.set_title(fig_title)

    width = 0.25
    iterations: np.ndarray = np.array([])  # Just to prevent MyPy warning
    for idx, worker in enumerate(data.keys()):
        np_array = np.array(data[worker])
        iterations = np_array[:, 0] + 1  # +1 to start from 1 instead of 0
        data_times_per_iteration = np_array[:, 1]
        # print(f'Worker {worker} | {data_type} times: {data_times_per_iteration}')
        margin = width * idx
        plt.bar(iterations + margin, data_times_per_iteration, width=width, label=worker)

    # Sets 10 as max value for y axis
    plt.ylim(0, 10)
    plt.xticks(iterations)

    # Gets labels and shows them sorted
    _handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(sorted(labels))


def __get_worker_id(worker_id: int) -> str:
    """Gets the worker identifier from the id passed by param (useful to standardize delays)."""
    return f'worker_{worker_id}'


def main():
    rdds: list[Rdd] = []
    execution_times_worker: WorkerBarTimes = {}
    idle_times_worker: WorkerBarTimes = {}

    previous_delay_percentage: Optional[DelayForWorker] = None

    for iteration in range(ITERATIONS):
        print(f'Iteration {iteration + 1}')
        print('====================================')

        for i in range(N_STARS):
            random_seed = (RANDOM_SEED + i) * (iteration + 1) if RANDOM_SEED is not None else None
            random_features_to_select = get_random_subset_of_features(N_FEATURES, random_state=random_seed)
            rdd = Rdd(i, random_features_to_select)
            rdds.append(rdd)

        if DEBUG:
            print('rdds before assign partitions')
            print(rdds)

        # Assigns the partition to each RDD
        rdds, partition_to_worker = __assign_partitions(rdds, STRATEGY, previous_delay_percentage)

        if DEBUG:
            print('rdds after assign partitions')
            print(rdds)

        # Generates a dict with the worker identifier as key and the delay to apply in every execution as value
        if ADD_DELAY_TO_WORKER_2:
            delay_for_worker: Optional[DelayForWorker] = {
                __get_worker_id(2): 0.75,
            }
        else:
            delay_for_worker = None

        # Executes the simulation
        current_execution_times, current_idle_times, previous_delay_percentage = collect(
            rdds,
            partition_to_worker,
            delay_for_worker
        )

        if DEBUG:
            print(f'worker_execution_times: {current_execution_times}')
            print(f'idle_times: {current_idle_times}')
            print(f'previous_delay_percentage: {previous_delay_percentage}')

        # Adds the data to execution and idle times
        __add_to_worker_dict(current_execution_times, execution_times_worker, iteration)
        __add_to_worker_dict(current_idle_times, idle_times_worker, iteration)

    generate_bar_charts(execution_times_worker, f'Strategy = {STRATEGY}', 'Execution')
    generate_bar_charts(idle_times_worker, f'Strategy = {STRATEGY}', 'Idle')
    plt.show()


if __name__ == '__main__':
    main()
