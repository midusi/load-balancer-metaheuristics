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
    print(f'Profits: {profits}')
    print(f'Weights: {weights}')
    print(f'Weights sum: {sum(weights)}')
    print(f'Capacities: {capacities}')


worker_to_delay: Optional[int] = None


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

            if worker_to_delay is not None:
                delay = random.randint(0, 1)
                for idx in range(len(capacities)):
                    if idx == worker_to_delay:
                        capacities[idx] -= capacities[idx] * delay
                    else:
                        # The other workers will have more capacity
                        capacities[idx] += capacities[idx] * delay // (N_WORKERS - 1)

            __print_all(tasks, weights, capacities)

            # Assign items into the knapsacks while maximizing profits
            start = time.time()
            res = solve_multiple_knapsack(tasks, weights, capacities, method='mthm', method_kwargs={'check_inputs': 0})
            # res = solve_multiple_knapsack(tasks, weights, capacities, method='mtm')
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

        print(worker_tasks_weights)
        sums = {k: sum(values) for k, values in worker_tasks_weights.items()}
        print(sums)
        print('=======================================')


if __name__ == '__main__':
    main()
