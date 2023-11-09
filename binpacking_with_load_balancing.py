import random
import time
from math import ceil
from typing import Final, Optional

from mknapsack import solve_multiple_knapsack

# All have a profit of 1. The weight is done randomly and one of the 3 knapsacks is selected randomly.
# to have a random backlog

N_ELEMENTS: Final = 1000
N_WORKERS: Final = 3


def __print_all(profits, weights, capacities):
    # print(f'Profits: {profits}')
    print(f'Weights: {weights}')
    print(f'Weights sum: {sum(weights)}')
    print(f'Capacities: {capacities}')


# WORKER_TO_DELAY: Final[Optional[int]] = None
WORKER_TO_DELAY: Final[Optional[int]] = 1


def main():
    for iteration in range(10):
        worker_tasks_weights = {
            1: [],
            2: [],
            3: []
        }
        print(f'Iteration {iteration}')
        print('=======================================')

        # Generates an array of 1 profit for every element (N_ELEMENTS)
        tasks = [1 for _ in range(N_ELEMENTS)]

        # Generates an array of random weights (float) for every element (N_ELEMENTS)
        weights = [ceil(random.uniform(0, 100)) for _ in range(N_ELEMENTS)]

        while len(tasks) > 0:
            # ...and N_WORKERS knapsacks with the same capacity (the sum of all the weights) / N_WORKERS
            weight_per_worker_equal = ceil(sum(weights) / N_WORKERS)
            min_weight = min(weights)
            weight_per_worker = max(weight_per_worker_equal, min_weight)

            capacities = [weight_per_worker for _ in range(N_WORKERS)]

            print('Before applying delay:')
            __print_all(tasks, weights, capacities)

            if WORKER_TO_DELAY is not None:
                delay_for_worker = random.uniform(0, 1)
                delay_for_the_rest = delay_for_worker / (N_WORKERS - 1)

                print(f'Updating capacities for a delay of {round(delay_for_worker * 100, 2)}%...')
                for idx in range(N_WORKERS):
                    old_capacity = capacities[idx]
                    if idx == WORKER_TO_DELAY:
                        capacities[idx] -= capacities[idx] * delay_for_worker
                    else:
                        # The other workers will have more capacity
                        capacities[idx] += capacities[idx] * delay_for_the_rest

                    capacities[idx] = ceil(capacities[idx])
                    print(
                        f'Worker {idx + 1}: {old_capacity} -> {capacities[idx]}. Incremented/Decremented in a {delay_for_the_rest * 100}%')

            __print_all(tasks, weights, capacities)

            # Assign items into the knapsacks while maximizing profits
            # NOTE: check_inputs=0 is used to avoid checking the inputs in the Fortran code and raise an error
            # when the number of task is less than the number of workers
            start = time.time()
            res = solve_multiple_knapsack(tasks, weights, capacities, method='mthm', method_kwargs={'check_inputs': 0})
            end = time.time()
            print(f'solve_multiple_knapsack() elapsed time: {end - start}')

            tasks_to_remove = set()
            for idx, bin_number in enumerate(res):
                # Bin_number == 0 means that the item was not assigned to any knapsack
                if bin_number != 0:
                    worker_tasks_weights[bin_number].append(weights[idx])  # Assigns the task to the worker

                    # Removes the task from the list of tasks and weights
                    tasks_to_remove.add(idx)

            tasks = [task for idx, task in enumerate(tasks) if idx not in tasks_to_remove]
            weights = [weight for idx, weight in enumerate(weights) if idx not in tasks_to_remove]

            if len(tasks) > 0:
                print(f'Backlog: {len(tasks)}')

        sums = {k: sum(values) for k, values in worker_tasks_weights.items()}
        print(f'Worker tasks weights: {sums}')
        print('=======================================')


if __name__ == '__main__':
    main()
