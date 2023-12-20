import pandas as pd
from typing import Literal
from simulator import ITERATIONS


def __get_time_type(df_n_stars_iteration: pd.DataFrame, df_binpacking_iteration: pd.DataFrame,
                    df_smart_iteration: pd.DataFrame, df_result: dict, data_type: Literal['execution', 'idle']):
    # Filters by the column 'type' == 'execution' and retrieves the mean and std
    df_n_stars_iteration_execution = df_n_stars_iteration[df_n_stars_iteration['type'] == data_type]
    df_binpacking_iteration_execution = df_binpacking_iteration[df_binpacking_iteration['type'] == data_type]
    df_smart_iteration_execution = df_smart_iteration[df_smart_iteration['type'] == data_type]

    # Retrieves the mean and std
    mean_n_stars = df_n_stars_iteration_execution['time'].mean()
    std_n_stars = df_n_stars_iteration_execution['time'].std()
    mean_binpacking = df_binpacking_iteration_execution['time'].mean()
    std_binpacking = df_binpacking_iteration_execution['time'].std()
    mean_smart = df_smart_iteration_execution['time'].mean()
    std_smart = df_smart_iteration_execution['time'].std()

    # Retrieves the strategy with the minimum mean
    if mean_smart < mean_binpacking and mean_smart < mean_n_stars:
        strategy_min = 'smart'
    elif mean_binpacking < mean_smart and mean_binpacking < mean_n_stars:
        strategy_min = 'binpacking'
    else:
        strategy_min = 'n_stars'

    # Appends the data to the result dataframe
    df_result[f'mean_n_stars_{data_type}'].append(mean_n_stars)
    df_result[f'std_n_stars_{data_type}'].append(std_n_stars)
    df_result[f'mean_binpacking_{data_type}'].append(mean_binpacking)
    df_result[f'std_binpacking_{data_type}'].append(std_binpacking)
    df_result[f'mean_smart_{data_type}'].append(mean_smart)
    df_result[f'std_smart_{data_type}'].append(std_smart)
    df_result[f'strategy_min_{data_type}'].append(strategy_min)


def main(random_seed: int):
    df_n_stars = pd.read_csv(f'Simulator_results/data_n_stars_random_{random_seed}.csv')
    df_binpacking = pd.read_csv(f'Simulator_results/data_binpacking_random_{random_seed}.csv')
    df_smart = pd.read_csv(f'Simulator_results/data_smart_random_{random_seed}.csv')
    df_result = {
        'iteration': [],
        'mean_n_stars_execution': [],
        'std_n_stars_execution': [],
        'mean_binpacking_execution': [],
        'std_binpacking_execution': [],
        'mean_smart_execution': [],
        'std_smart_execution': [],
        'mean_n_stars_idle': [],
        'std_n_stars_idle': [],
        'mean_binpacking_idle': [],
        'std_binpacking_idle': [],
        'mean_smart_idle': [],
        'std_smart_idle': [],
        'strategy_min_execution': [],
        'strategy_min_idle': [],
    }

    # Iterates over the number of iterations and retrieves all the data from the dataframes
    for iteration in range(ITERATIONS):
        df_n_stars_iteration = df_n_stars[df_n_stars['iteration'] == iteration]
        df_binpacking_iteration = df_binpacking[df_binpacking['iteration'] == iteration]
        df_smart_iteration = df_smart[df_smart['iteration'] == iteration]

        # Adds iteration
        df_result['iteration'].append(iteration + 1)  # Starts at 1 like the bar plots

        # Retrieves the mean and std for the execution and idle time for each strategy
        __get_time_type(df_n_stars_iteration, df_binpacking_iteration, df_smart_iteration, df_result, 'execution')
        __get_time_type(df_n_stars_iteration, df_binpacking_iteration, df_smart_iteration, df_result, 'idle')

    # Creates the dataframe with the results
    df_result = pd.DataFrame(df_result)
    print(df_result)

    # Saves the dataframe to a csv file
    df_result.to_csv(f'Simulator_results/comparison_result_random_{random_seed}.csv', index=False)


if __name__ == '__main__':
    for random_seed_to_test in range(100, 1000, 100):
        print(f'Random seed: {random_seed_to_test}')
        print('====================================')
        main(random_seed=random_seed_to_test)
