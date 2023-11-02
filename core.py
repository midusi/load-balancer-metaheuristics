import json
import logging
import os
import re
import time
import numpy as np
import pandas as pd
from pyspark import SparkContext, Broadcast
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List, Optional, Callable, Union, Tuple, Any, Dict, cast

from load_balancer_parameters import SVMParameters, ClusteringParameters, RFParameters
from metaheuristics import binary_black_hole, improved_binary_black_hole, binary_black_hole_spark, \
    CrossValidationSparkResult, parallelize_fitness_execution_by_partitions
from utils import get_columns_from_df, read_survival_data, DatasetName, ModelName

logging.getLogger().setLevel(logging.INFO)

# Prevents 'A value is trying to be set on a copy of a slice from a DataFrame.' error
pd.options.mode.chained_assignment = None

# Prevents 'A value is trying to be set on a copy of a slice from a DataFrame.' error
pd.options.mode.chained_assignment = None

# Some useful types
ParameterFitnessFunctionSequential = Tuple[pd.DataFrame, np.ndarray]
ParsedDataCallable = Callable[[np.ndarray, Any, np.ndarray], Union[ParameterFitnessFunctionSequential, SparkDataFrame]]

# Fitness function result structure. It's a function that takes a Pandas DF/Spark Broadcast variable, the bool subset
# of features, the original data (X), the target vector (Y) and a bool flag indicating if it's a broadcast variable
CrossValidationCallback = Callable[[Union[pd.DataFrame, Broadcast], np.ndarray, np.ndarray, bool],
                                   CrossValidationSparkResult]

# Fitness function only for sequential experiments
CrossValidationCallbackSequential = Callable[[pd.DataFrame, np.ndarray], CrossValidationSparkResult]


def fitness_function_with_checking(
        compute_cross_validation: CrossValidationCallback,
        index_array: np.ndarray,
        x: Union[pd.DataFrame, Broadcast],
        y: np.ndarray,
        is_broadcast: bool
) -> CrossValidationSparkResult:
    """
    Fitness function of a star evaluated in the Binary Black hole, including vector without features check.

    :param compute_cross_validation: Fitness function
    :param index_array: Boolean vector to indicate which feature will be present in the fitness function
    :param x: Data with features
    :param y: Classes
    :param is_broadcast: True if x is a Spark Broadcast to retrieve its values
    :return: All the results, documentation listed in the CrossValidationSparkResult type
    """
    if not np.count_nonzero(index_array):
        return -1.0, -1.0, -1, '', -1, '', -1.0, -1.0, -1.0, -1.0

    return compute_cross_validation(x, index_array, y, is_broadcast)


def run_experiment(
        app_name: str,
        compute_cross_validation: Union[CrossValidationCallback, CrossValidationCallbackSequential],
        metric_description: str,
        model_name: ModelName,
        parameters_description: str,
        add_epsilon: bool,
        dataset: DatasetName,
        number_of_independent_runs: int,
        n_iterations: int,
        number_of_workers: int,
        more_is_better: bool,
        n_stars: int,
        random_state: Optional[int],
        run_improved_bbha: Optional[bool] = None,
        load_balancer_parameters: Optional[Union[SVMParameters, ClusteringParameters, RFParameters]] = None,
        run_in_spark: bool = False,
        debug: bool = False,
        sc: Optional[SparkContext] = None,
        coeff_1: float = 2.35,
        coeff_2: float = 0.2,
        binary_threshold: Optional[float] = None,
        use_broadcasts_in_spark: Optional[bool] = True
):
    """
    Computes the BBHA and/or Improved BBHA algorithm/s

    :param app_name: App name to save the CSV result and all the execution metrics.
    :param compute_cross_validation: Cross Validation function to get the fitness.
    :param metric_description: Description of the metric returned by the CrossValidation function to display in the CSV.
    :param model_name: Description of the model used as CrossValidation fitness function to be displayed in the CSV.
    :param parameters_description: Model's parameters description to report in results.
    :param add_epsilon: If True it adds an epsilon to 0s in Y data to prevent errors in SVM training.
    :param dataset: Dataset name to use (name of the sub folder in the 'Datasets' folder).
    :param number_of_workers: Number of workers nodes in the Spark cluster.
    :param more_is_better: If True, it returns the highest value (SVM and RF C-Index), lowest otherwise (LogRank p-value).
    :param load_balancer_parameters: Parameters to train the load balancer model.
    :param run_improved_bbha: If None runs both algorithm versions. True for improved, False to run the original.
    :param run_in_spark: True to run the stars of the BBHA in a distributed Apache Spark cluster.
    :param debug: True to log extra data during script execution.
    :param sc: Spark Context. Only used if 'run_in_spark' = True.
    :param number_of_independent_runs: Number of independent runs. On every independent run it stores a JSON file with.
    data from the BBHA execution. This parameter is NOT the number of iterations inside the BBHA algorithm.
    :param n_stars: Number of stars in the BBHA algorithm.
    :param random_state: Random state to replicate experiments. It allows to set the same number of features for every
    star and the same shuffling.
    :param n_iterations: Number of iterations in the BBHA algorithm.
    :param coeff_1: Coefficient 1 required by the enhanced version of the BBHA algorithm.
    :param coeff_2: Coefficient 2 required by the enhanced version of the BBHA algorithm.
    :param binary_threshold: Threshold used in BBHA, None to be computed randomly.
    :param use_broadcasts_in_spark: If True, it generates a Broadcast value to pass to the fitness function instead of pd.DataFrame. Is ignored if run_in_spark = False.
    """
    if run_in_spark and number_of_workers == 0:
        logging.error(f'Invalid number of workers in Spark Cluster ({number_of_workers}). '
                      'Check "number_of_workers" parameter or set "run_in_spark" = False!')
        return

    # CSV where the results will be stored
    now = time.strftime('%Y-%m-%d_%H_%M_%S')
    current_script_dir_name = os.path.dirname(__file__)

    # Configures CSV file
    app_folder = f'Results/{app_name}'
    res_csv_file_path = os.path.join(current_script_dir_name, f'{app_folder}/result_{now}.csv')

    logging.info(f'Metaheuristic results will be saved in "{res_csv_file_path}"')
    logging.info(f'Metrics will be saved in JSON files (one per iteration) inside folder "{app_folder}"')

    # Creates a folder to save all the results and figures
    mode = 0o777
    dir_path = os.path.join(current_script_dir_name, app_folder)
    os.mkdir(dir_path, mode)
    os.chmod(dir_path, mode)  # Mode in mkdir is sometimes ignored: https://stackoverflow.com/a/5231994/7058363

    best_metric_with_all_features = f'{metric_description} with all the features'
    best_metric_in_runs_key = f'Best {metric_description} (in {number_of_independent_runs} runs)'
    res_csv = pd.DataFrame(columns=['dataset', 'Improved BBHA', 'Model',
                                    best_metric_with_all_features,
                                    best_metric_in_runs_key,
                                    f'Features with best {metric_description} (in {number_of_independent_runs} runs)',
                                    f'CPU execution time ({number_of_independent_runs} runs) in seconds'])

    # Gets survival data
    x, y = read_survival_data(add_epsilon, dataset_folder=dataset)

    number_samples, number_features = x.shape

    logging.info(f'Running {number_of_independent_runs} independent runs of the BBHA experiment with {n_iterations} '
                 f'iterations and {n_stars} stars')
    if run_in_spark:
        logging.info(f'Running {n_stars} stars in Spark ({n_stars // number_of_workers} stars per worker). '
                     f'{number_of_workers} active workers in Spark Cluster')
        load_balancer_desc = 'With' if load_balancer_parameters is not None else 'Without'
        logging.info(f'{load_balancer_desc} load balancer')

    logging.info(f'Metric: {metric_description} | Model: {model_name} | '
                 f'Parameters: {parameters_description} | Random state: {random_state}')
    logging.info(f'Survival dataset: "{dataset}"')
    logging.info(f'\tSamples (rows): {number_samples} | Features (columns): {number_features}')
    logging.info(f'\tY shape: {y.shape[0]}')

    # If it was set, generates a broadcast value
    using_broadcast = run_in_spark and use_broadcasts_in_spark
    if using_broadcast:
        logging.info('Using Broadcast')
        x = sc.broadcast(x)

    # Gets concordance index with all the features
    start = time.time()
    cv_result = compute_cross_validation(x, np.ones(number_features), y, use_broadcasts_in_spark)
    all_features_concordance_index = cv_result[0]  # It's the first element

    logging.info(f'Fitness function with all the features finished in {time.time() - start} seconds')
    logging.info(f'{metric_description} with all the features: {all_features_concordance_index}')

    # Check which version of the algorithm want to run
    if run_improved_bbha is None:
        improved_options = [False, True]
    elif run_improved_bbha is True:
        improved_options = [True]
    else:
        improved_options = [False]

    experiment_start = time.time()
    for run_improved in improved_options:
        improved_mode_str = 'improved' if run_improved else 'normal'
        spark_mode_str = '(in Spark)' if run_in_spark else '(sequential)'
        logging.info(f'Running {improved_mode_str} algorithm {spark_mode_str}')
        independent_start_time = time.time()

        final_subset = None  # Final best subset
        best_metric = -1 if more_is_better else 99999  # Final best metric

        for independent_run_i in range(number_of_independent_runs):
            logging.info(f'Independent run {independent_run_i + 1}/{number_of_independent_runs}')

            # Binary Black Hole
            bh_start = time.time()
            if run_improved:
                json_experiment_data = {}  # No Spark, no data about execution times to store
                best_subset, current_metric = improved_binary_black_hole(
                    n_stars=n_stars,
                    n_features=number_features,
                    n_iterations=n_iterations,
                    fitness_function=lambda subset: fitness_function_with_checking(
                        compute_cross_validation,
                        subset,
                        x,
                        y,
                        is_broadcast=using_broadcast
                    ),
                    coeff_1=coeff_1,
                    coeff_2=coeff_2,
                    binary_threshold=binary_threshold,
                    debug=debug
                )
            else:
                if run_in_spark:
                    # Sets the number of samples to use the load balancer
                    if load_balancer_parameters is not None:
                        load_balancer_parameters.number_of_samples = number_samples

                    best_subset, current_metric, _best_data, json_experiment_data = binary_black_hole_spark(
                        n_stars=n_stars,
                        n_features=number_features,
                        n_iterations=n_iterations,
                        fitness_function=lambda subset: fitness_function_with_checking(
                            compute_cross_validation,
                            subset,
                            x,
                            y,
                            is_broadcast=using_broadcast
                        ),
                        sc=sc,
                        binary_threshold=binary_threshold,
                        more_is_better=more_is_better,
                        random_state=random_state,
                        debug=debug,
                        number_of_workers=number_of_workers,
                        load_balancer_parameters=load_balancer_parameters,
                    )
                else:
                    best_subset, current_metric, json_experiment_data = binary_black_hole(
                        n_stars=n_stars,
                        n_features=number_features,
                        n_iterations=n_iterations,
                        fitness_function=lambda subset: fitness_function_with_checking(
                            compute_cross_validation,
                            subset,
                            x,
                            y,
                            is_broadcast=using_broadcast
                        ),
                        random_state=random_state,
                        binary_threshold=binary_threshold,
                        debug=debug
                    )

            iteration_time = time.time() - bh_start
            logging.info(f'Independent run {independent_run_i + 1}/{number_of_independent_runs} | '
                         f'Binary Black Hole with {n_iterations} iterations and {n_stars} '
                         f'stars, finished in {iteration_time} seconds')

            # Check if current is the best metric
            if (more_is_better and current_metric > best_metric) or (not more_is_better and current_metric < best_metric):
                best_metric = current_metric

                # Gets columns names
                x_df = x.value if run_in_spark and use_broadcasts_in_spark else x
                column_names = get_columns_from_df(best_subset, x_df).columns.values
                final_subset = column_names

            # Stores data to train future load balancer models
            # Adds data to JSON
            json_extra_data = {
                'model': model_name,
                'dataset': dataset,
                'parameters': parameters_description,
                'number_of_samples': number_samples,
                'independent_iteration_time': iteration_time
            }
            json_experiment_data = {**json_experiment_data, **json_extra_data}

            now = time.strftime('%Y-%m-%d_%H_%M_%S')
            json_file = f'{model_name}_{parameters_description}_{metric_description}_{dataset}_{now}_' \
                        f'iteration_{independent_run_i}_results.json'
            json_file = re.sub(' +', '_', json_file).lower()  # Replaces whitespaces with '_' and makes lowercase
            json_dest = os.path.join(app_folder, json_file)

            with open(json_dest, 'w+') as file:
                file.write(json.dumps(json_experiment_data))

        # Reports final Feature Selection result
        independent_run_time = round(time.time() - independent_start_time, 3)
        logging.info(f'{number_of_independent_runs} independent runs finished in {independent_run_time} seconds')

        experiment_results_dict = {
            'dataset': dataset,
            'Improved BBHA': 1 if run_improved else 0,
            'Model': model_name,
            best_metric_with_all_features: round(all_features_concordance_index, 4),
            best_metric_in_runs_key: round(best_metric, 4),
            f'Features with best {metric_description} (in {number_of_independent_runs} runs)': ' | '.join(final_subset),
            f'CPU execution time ({number_of_independent_runs} runs) in seconds': independent_run_time
        }

        # Some extra reporting
        algorithm = 'BBHA' + (' (improved)' if run_improved else '')
        logging.info(f'Found {len(final_subset)} features with {algorithm} ({metric_description} '
                     f'= {best_metric}):')
        logging.info(final_subset)

        # Saves new data to final CSV
        res_csv = pd.concat([res_csv, pd.DataFrame([experiment_results_dict])], ignore_index=True)
        res_csv.to_csv(res_csv_file_path)

    logging.info(f'Experiment completed in {time.time() - experiment_start} seconds')


def run_times_experiment(
        app_name: str,
        compute_cross_validation: CrossValidationCallback,
        n_iterations: int,
        number_of_independent_runs: int,
        n_stars: int,
        metric_description: str,
        model_name: ModelName,
        parameters_description: str,
        add_epsilon: bool,
        dataset: DatasetName,
        number_of_workers: int,
        load_balancer_parameters: Optional[Union[SVMParameters, ClusteringParameters, RFParameters]] = None,
        sc: Optional[SparkContext] = None,
        use_broadcasts_in_spark: Optional[bool] = True,
        debug: bool = True
):
    """
    Runs an experiment to get a lot of metrics with different number of features.
    :param app_name: App name to save the CSV result and all the execution metrics.
    :param n_iterations: Number of iterations to simulate in the BBHA. There will be saved the metrics for every
    amount features (10, 20, ..., etc.) the same amount of times that this parameter.
    :param number_of_independent_runs: Number of independent runs. On every independent run it stores a JSON file with
    the 'n_iterations' entries for every amount of features.
    :param n_stars: Number of stars
    :param compute_cross_validation: Fitness function
    :param metric_description: Metric description to report in results
    :param model_name: Model name to report in results
    :param parameters_description: Model's parameters description to report in results
    :param add_epsilon: If True it adds an epsilon to 0s in Y data to prevent errors in SVM training
    :param dataset: Dataset name to use (name of the sub folder in the 'Datasets' folder)
    :param number_of_workers: Number of workers nodes in the Spark cluster
    :param load_balancer_parameters: Parameters to train the load balancer model.
    :param sc: Spark Context
    :param use_broadcasts_in_spark: If True, it generates a Broadcast value to pass to the fitness function instead of pd.DataFrame. Is ignored if run_in_spark = False
    :param debug: If True it logs all the star values in the terminal
    """
    if number_of_workers == 0:
        logging.error(f'Invalid number of workers in Spark Cluster ({number_of_workers}). '
                      f'Check "number_of_workers" parameter!')
        return

    # Creates a folder to save all the results and figures
    current_script_dir_name = os.path.dirname(__file__)

    app_folder = f'Times results/{app_name}'
    mode = 0o777
    dir_path = os.path.join(current_script_dir_name, app_folder)
    os.mkdir(dir_path, mode)
    os.chmod(dir_path, mode)  # Mode in mkdir is sometimes ignored: https://stackoverflow.com/a/5231994/7058363

    logging.info(f'Metrics will be saved in JSON files (one per independent iteration) inside folder "{app_folder}"')

    # Gets survival data
    x, y = read_survival_data(add_epsilon, dataset_folder=dataset)

    number_samples, number_features = x.shape
    step = 100

    # Some informative logs
    logging.info(f'Running {number_of_independent_runs} independent runs of the times experiment with {n_iterations} '
                 f'iterations and {n_stars} stars ({n_stars // number_of_workers} stars per worker). '
                 f'{number_of_workers} active workers in Spark Cluster')

    load_balancer_desc = 'With' if load_balancer_parameters is not None else 'Without'
    logging.info(f'{load_balancer_desc} load balancer')

    logging.info(f'Metric: {metric_description} | Model: {model_name} | '
                 f'Parameters: {parameters_description}')
    logging.info(f'Survival dataset: "{dataset}"')
    logging.info(f'\tSamples (rows): {number_samples} | Features (columns): {number_features}')
    logging.info(f'\tY shape: {y.shape[0]}')

    # Needed parameter for the Binary Black Hole Algorithm
    total_n_features = number_features

    if use_broadcasts_in_spark:
        initial_n_features = 1000

        logging.info(f'Broadcasting enabled. Running initial experiment with {n_stars} stars with {initial_n_features} '
                     f'features to broadcast data to all the workers...')
        x = sc.broadcast(x)

        # Runs an initial experiment with 1000 features to broadcast the data and prevent issues with execution times
        # due to data distribution
        stars_subsets_initial = np.empty((n_stars, 2), dtype=object)  # 2 = (1, features)
        for i in range(n_stars):
            random_features_to_select_initial = np.zeros(total_n_features, dtype=int)
            random_features_to_select_initial[:initial_n_features] = 1
            np.random.shuffle(random_features_to_select_initial)
            stars_subsets_initial[i] = (i + 1, random_features_to_select_initial)

        _results_values, total_initial_time, _predicted_times = parallelize_fitness_execution_by_partitions(
            sc,
            stars_subsets_initial,
            fitness_function=lambda subset: fitness_function_with_checking(
                compute_cross_validation,
                subset,
                x,
                y,
                is_broadcast=use_broadcasts_in_spark
            ),
            load_balancer_parameters=None,  # Load balancer is not used in this stage
            number_of_workers=number_of_workers,
            debug=debug
        )
        logging.info(f'Initial running with {initial_n_features} features finished in {total_initial_time}')
    else:
        logging.info(f'Broadcasting disabled. Initial run discarded ')

    for independent_run_i in range(number_of_independent_runs):
        logging.info(f'Independent run {independent_run_i + 1}/{number_of_independent_runs}')

        # Lists for storing times and fitness data to train models
        number_of_features: List[int] = []
        hosts: List[str] = []
        partition_ids: List[int] = []
        fitness: List[float] = []
        time_exec: List[float] = []
        predicted_time_exec: List[float] = []
        times_by_iteration: List[float] = []
        time_test: List[float] = []
        num_of_iterations: List[float] = []
        train_scores: List[float] = []

        # To report every Spark's Worker idle time
        workers_idle_times: Dict[str, List[Tuple[int, float]]] = {}
        workers_execution_times_per_iteration: Dict[str, List[Tuple[int, float]]] = {}

        # Runs the iterations
        for i_iter in range(n_iterations):
            logging.info(f'Iteration {i_iter + 1}/{n_iterations}')

            stars_subsets = np.empty((n_stars, 2), dtype=object)  # 2 = (1, features)
            current_n_features = 10

            while current_n_features <= total_n_features:
                # To report every Spark's Worker idle time. It stores the execution time for the current iteration
                workers_execution_times: Dict[str, float] = {}

                for i in range(n_stars):
                    # Initializes 'Population' with a key for partitionBy()
                    random_features_to_select = np.zeros(total_n_features, dtype=int)
                    random_features_to_select[:current_n_features] = 1
                    np.random.shuffle(random_features_to_select)
                    stars_subsets[i] = (i, random_features_to_select)

                    # Jumps by 10 elements, after 100 features it jumps by 'step' elements
                    if current_n_features < step:
                        current_n_features += 10
                    else:
                        current_n_features += step

                    # If it's raised the maximum number of features, slices the stars array
                    if current_n_features > total_n_features:
                        stars_subsets = stars_subsets[:i + 1]
                        break

                results_values, total_iteration_time, predicted_times_map = parallelize_fitness_execution_by_partitions(
                    sc,
                    stars_subsets,
                    fitness_function=lambda subset: fitness_function_with_checking(
                        compute_cross_validation,
                        subset,
                        x,
                        y,
                        is_broadcast=use_broadcasts_in_spark
                    ),
                    load_balancer_parameters=load_balancer_parameters,
                    number_of_workers=number_of_workers,
                    debug=debug
                )

                for star_idx, current_data in results_values:
                    current_fitness_mean = current_data[0]
                    worker_execution_time = current_data[1]
                    partition_id = current_data[2]
                    host_name = current_data[3]
                    evaluated_features = current_data[4]
                    time_lapse_description = current_data[5]
                    time_by_iteration = current_data[6]
                    model_test_time = current_data[7]
                    mean_num_of_iterations = current_data[8]
                    train_score = current_data[9]
                    current_predicted_time = predicted_times_map[star_idx]

                    number_of_features.append(evaluated_features)
                    hosts.append(host_name)
                    partition_ids.append(partition_id)
                    fitness.append((round(current_fitness_mean, 4)))
                    time_exec.append(round(worker_execution_time, 4))
                    times_by_iteration.append(round(time_by_iteration, 4))
                    time_test.append(round(model_test_time, 4))
                    num_of_iterations.append(round(mean_num_of_iterations, 4))
                    train_scores.append(round(train_score, 4))
                    predicted_time_exec.append(round(current_predicted_time, 4))

                    # Adds execution times to compute the idle time for all the workers. It's the difference between the
                    # time it took to the master to distribute all the computations and get the result from all the workers;
                    # and the sum of all the execution times for every star every Worker got
                    if host_name not in workers_execution_times:
                        workers_execution_times[host_name] = 0.0
                    workers_execution_times[host_name] += worker_execution_time

                    if debug:
                        logging.info(f'{star_idx} star took {round(worker_execution_time, 3)} seconds ({time_lapse_description}) '
                                     f'for {evaluated_features} features. Partition: {partition_id} | '
                                     f'Host name: {host_name}. Fitness: {current_fitness_mean}')

                # Stores the idle time for every worker in this iteration
                for host_name, sum_execution_times in workers_execution_times.items():
                    if host_name not in workers_idle_times:
                        workers_idle_times[host_name] = []

                    if host_name not in workers_execution_times_per_iteration:
                        workers_execution_times_per_iteration[host_name] = []

                    if debug:
                        logging.info(f'The worker {host_name} has taken ~{sum_execution_times} seconds to compute all its stars')

                    workers_execution_times_per_iteration[host_name].append((i_iter, sum_execution_times))
                    workers_idle_times[host_name].append((i_iter, total_iteration_time - sum_execution_times))

        # Saves times in JSON for post-processing
        now = time.strftime('%Y-%m-%d_%H_%M_%S')
        json_file = f'{model_name}_{parameters_description}_{metric_description}_{dataset}_{now}_' \
                    f'iteration_{independent_run_i}_results.json'
        json_file = re.sub(' +', '_', json_file).lower()  # Replaces whitespaces with '_' and makes lowercase
        json_dest = os.path.join(app_folder, json_file)

        # Computes avg/std of idle times for all the iterations for every Worker
        workers_idle_times_res: Dict[str, Dict[str, float]] = {}
        for host_name, idle_times in workers_idle_times.items():
            workers_idle_times_res[host_name] = {
                'mean': round(cast(float, np.mean(idle_times)), 4),
                'std': round(cast(float, np.std(idle_times)), 4)
            }

        logging.info(f'Saving lists in JSON format in {json_dest}')
        result_dict = {
            'number_of_features': number_of_features,
            'execution_times': time_exec,
            'predicted_execution_times': predicted_time_exec,
            'fitness': fitness,
            'times_by_iteration': times_by_iteration,
            'test_times': time_test,
            'train_scores': train_scores,
            'number_of_iterations': num_of_iterations,
            'hosts': hosts,
            'workers_execution_times_per_iteration': workers_execution_times_per_iteration,
            'workers_idle_times': workers_idle_times_res,  # Yes, names are confusing, but workers_idle_times_res has mean and std
            'workers_idle_times_per_iteration': workers_idle_times,
            'partition_ids': partition_ids,
            'model': model_name,
            'dataset': dataset,
            'parameters': parameters_description,
            'number_of_samples': number_samples
        }

        with open(json_dest, 'w+') as file:
            file.write(json.dumps(result_dict))

    logging.info('Saved.')


def fitness_function_with_checking_sequential(
        compute_cross_validation: CrossValidationCallbackSequential,
        index_array: np.array,
        x: Union[pd.DataFrame, Broadcast],
        y: np.ndarray,
) -> CrossValidationSparkResult:
    """
    Fitness function of a star evaluated in the Binary Black hole, including featureless vector check for sequential
    experiment.
    :param compute_cross_validation: Fitness function.
    :param index_array: Boolean vector to indicate which feature will be present in the fitness function
    :param x: Data with features
    :param y: Classes
    :return: All the results, documentation listed in the CrossValidationSparkResult type
    """
    if not np.count_nonzero(index_array):
        return -1.0, -1.0, -1, '', -1, '', -1.0, -1.0, -1.0, -1.0

    parsed_data = get_columns_from_df(index_array, x)
    return compute_cross_validation(parsed_data, y)


def run_times_experiment_sequential(
        compute_cross_validation: CrossValidationCallbackSequential,
        n_iterations: int,
        metric_description: str,
        model_name: ModelName,
        parameters_description: str,
        add_epsilon: bool
):
    """
    Same function as run_times_experiment but this is not distributed in Spark cluster. Everything run on same machine

    :param compute_cross_validation: Fitness function
    :param n_iterations: Number of iterations
    :param metric_description: Metric description to report in results
    :param model_name: Model description to report in results
    :param parameters_description: Model's parameters description to report in results
    :param add_epsilon: If True it adds an epsilon to 0s in Y data to prevent errors in SVM training
    """
    # Gets survival data
    x, y = read_survival_data(add_epsilon)

    number_samples, number_features = x.shape
    step = 100

    logging.info(f'Running times experiment with {n_iterations} iterations SEQUENTIALLY')
    logging.info(f'Metric: {metric_description} | Model: {model_name}')
    logging.info(f'Survival dataset')
    logging.info(f'\tSamples (rows): {number_samples} | Features (columns): {number_features}')
    logging.info(f'\tY shape: {y.shape[0]}')

    # Needed parameter for the Binary Black Hole Algorithm
    total_n_features = x.shape[1]

    # Lists for reporting
    number_of_features: List[int] = []
    exec_times: List[float] = []

    # Runs the iterations
    for i_iter in range(n_iterations):
        logging.info(f'Iteration {i_iter + 1}/{n_iterations}')

        current_n_features = step

        while current_n_features <= total_n_features:
            random_features_to_select = np.zeros(total_n_features, dtype=int)
            random_features_to_select[:current_n_features] = 1
            np.random.shuffle(random_features_to_select)

            start_worker_time = time.time()
            fitness_function_with_checking_sequential(
                compute_cross_validation,
                random_features_to_select,
                x,
                y
            )
            cur_exec_time = time.time() - start_worker_time

            number_of_features.append(current_n_features)
            exec_times.append(round(cur_exec_time, 4))

            current_n_features += step

    # Saves times in JSON for post-processing
    now = time.strftime('%Y-%m-%d_%H_%M_%S')
    json_file = f'{model_name}_{parameters_description}_{metric_description}_{now}_results_sequential.json'
    json_file = re.sub(' +', '_', json_file).lower()  # Replaces whitespaces with '_' and makes lowercase
    json_dest = os.path.join('Times results', json_file)

    logging.info(f'Saving lists in JSON format in {json_dest}')
    result_dict = {
        'n_features': number_of_features,
        'execution_times': exec_times,
    }

    with open(json_dest, 'w+') as file:
        file.write(json.dumps(result_dict))

    logging.info('Saved.')
