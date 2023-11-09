import random
import time
from math import ceil
from typing import Final, Optional

from mknapsack import solve_multiple_knapsack

# All have a profit of 1. The weight is done randomly and one of the 3 knapsacks is selected randomly.
# to have a random backlog

N_ELEMENTS: Final = 60
N_WORKERS: Final = 3


def __print_all(profits, weights, capacities):
    print(f'Profits: {profits}')
    print(f'Weights: {weights}')
    print(f'Weights sum: {sum(weights)}')
    print(f'Capacities: {capacities}')


worker_to_delay: Optional[int] = None

for iteration in range(10):
    worker_tasks = {
        1: [],
        2: [],
        3: []
    }
    print(f'Iteration {iteration}')
    print('=======================================')
    # Generates an array of 1 profit for every element (N_ELEMENTS)
    profits = [1 for _ in range(N_ELEMENTS)]

    # Generates an array of random weights (float) for every element (N_ELEMENTS)
    weights = [ceil(random.uniform(0, 100)) for _ in range(N_ELEMENTS)]

    # ...and N_WORKERS knapsacks with the same capacity (the sum of all the weights) / N_WORKERS
    capacities = [ceil(sum(weights) / N_WORKERS) for _ in range(N_WORKERS)]

    if worker_to_delay is not None:
        delay = random.randint(0, 1)
        for idx in range(len(capacities)):
            if idx == worker_to_delay:
                capacities[idx] -= capacities[idx] * delay
            else:
                # The other workers will have more capacity
                capacities[idx] += capacities[idx] * delay // (N_WORKERS - 1)

    __print_all(profits, weights, capacities)

    # Assign items into the knapsacks while maximizing profits
    start = time.time()
    res = solve_multiple_knapsack(profits, weights, capacities)
    end = time.time()
    print(f'Elapsed time: {end - start}')

    sums = {}
    
    for idx, bin_number in enumerate(res):
        # Bin_number == 0 means that the item was not assigned to any knapsack
        if bin_number == 0:
            raise Exception(f'Item {idx} was not assigned to any knapsack')

        if bin_number not in sums:
            sums[bin_number] = 0

        sums[bin_number] += weights[idx]

    print(sums)
    print('=======================================')