from typing import Tuple, Union, Literal, List
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from pyspark import TaskContext, Broadcast
from pyspark.sql import SparkSession
from sklearn.cluster import KMeans, SpectralClustering
from core import run_experiment, run_times_experiment, run_times_experiment_sequential
from load_balancer_parameters import SVMParameters, ClusteringParameters, RFParameters
from utils import get_columns_from_df, DatasetName, MASTER_CONNECTION_URL, ModelName, ClusteringAlgorithm, \
    ClusteringScoringMethod
import pandas as pd
from sklearn.model_selection import cross_validate
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastKernelSurvivalSVM
import logging
import time
from pyspark import SparkConf, SparkContext
from typing import Optional
import socket
from datetime import datetime
from multiprocessing import Process, Queue
from filelock import FileLock

# Enables info logging
logging.getLogger().setLevel(logging.INFO)

# If True, runs Feature Selection experiment. If False, runs times experiment
RUN_TIMES_EXPERIMENT: bool = False

# Number of cores used by the worker to compute the Cross Validation. -1 = use all
# This can be tricky since Spark assigns a certain number of cores to the executor, so you cannot be sure that it is
# always the number specified here, if possible set the same as CORES_PER_EXECUTOR.
N_JOBS = 1

# Number of folds to compute in the Cross Validation
NUMBER_OF_CV_FOLDS: int = 10

# To get the training score or not
RETURN_TRAIN_SCORE = True

# To replicate randomness
# RANDOM_STATE: Optional[int] = None
RANDOM_STATE: Optional[int] = 13

# If True, load balancer is used to generate Spark partitions
USE_LOAD_BALANCER: bool = False
# USE_LOAD_BALANCER: bool = True

# Number of independent complete runs to get the best parameters or number of experiments to compute all the times
# NUMBER_OF_INDEPENDENT_RUNS = 30
# NUMBER_OF_INDEPENDENT_RUNS = 2
NUMBER_OF_INDEPENDENT_RUNS = 1

# If True runs in Spark, otherwise in sequential.
# IMPORTANT: if this is True, set the parameter --driver-memory="6g" to avoid memory issues on sequential experiments
# RUN_IN_SPARK: bool = True
RUN_IN_SPARK: bool = False

# If True, runs the improved version of BBHA (only used if RUN_IN_SPARK is False, for the moment...). None to try
# both versions
RUN_IMPROVED_BBHA: Optional[bool] = False

# Identifier of the dataset to use
DATASET: DatasetName = 'Breast_Invasive_Carcinoma'

# Number of iterations for the BBHA algorithm
# N_ITERATIONS = 3
N_ITERATIONS = 2
# N_ITERATIONS = 30

# Number of stars for the BBHA algorithm
N_STARS = 3
# N_STARS = 15

# If True Random Forest is used as classificator. SVM otherwise
# MODEL_TO_USE: ModelName = 'clustering'
MODEL_TO_USE: ModelName = 'svm'
# MODEL_TO_USE: ModelName = 'rf'

# RF parameters
# Number of trees in Random Forest
RF_N_ESTIMATORS: int = 5

# SVM parameters
# If True, a regression task is performed, otherwise it executes a ranking task
IS_SVM_REGRESSION: bool = True

# Max number of SVM iterations
MAX_ITERATIONS: int = 1000

# SVM kernel function. NOTE: 'sigmoid' presents many NaNs values and 'precomputed' doesn't work in this context
SVM_KERNEL: Literal["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"] = 'linear'
SVM_OPTIMIZER: Literal["avltree", "rbtree"] = 'avltree'

# Clustering model to use
CLUSTERING_ALGORITHM: ClusteringAlgorithm = 'kmeans' # TODO: add more

# Number of clusters to group by molecule expressions during clustering algorithm
NUMBER_OF_CLUSTERS: int = 2

# Scoring method to use in the clustering algorithms
CLUSTERING_SCORING_METHOD: ClusteringScoringMethod = 'log_likelihood'

# To use a Broadcast value instead of a pd.DataFrame
USE_BROADCAST = True

# If True, means that fitness function returns a value that must be maximized. Otherwise, it must be minimized
MORE_IS_BETTER: bool = True  # For the moment all the metrics need to be maximized

# Only if RUN_IN_SPARK is set to True the following parameters are used
# Executors per instance of each worker
EXECUTORS: Optional[str] = "1"

# Cores on each executor
CORES_PER_EXECUTOR: Optional[str] = None

# RAM to use per executor
MEMORY_PER_EXECUTOR: str = "6g"

# If True it logs all the star values in the terminal
# DEBUG: bool = False
DEBUG: bool = True

# Classificator
if MODEL_TO_USE == 'rf':
    CLASSIFIER = RandomSurvivalForest(n_estimators=RF_N_ESTIMATORS,
                                      min_samples_split=10,
                                      min_samples_leaf=15,
                                      max_features="sqrt",
                                      n_jobs=N_JOBS,
                                      random_state=RANDOM_STATE)
elif MODEL_TO_USE == 'svm':
    rank_ratio = 0.0 if IS_SVM_REGRESSION else 1.0
    CLASSIFIER = FastKernelSurvivalSVM(rank_ratio=rank_ratio, max_iter=MAX_ITERATIONS, tol=1e-5, kernel=SVM_KERNEL,
                                       optimizer=SVM_OPTIMIZER, random_state=RANDOM_STATE)
else:
    CLASSIFIER = None


def get_clustering_model() -> Union[KMeans, SpectralClustering]:
    """Gets the specified clustering model to train"""
    if CLUSTERING_ALGORITHM == 'kmeans':
        return KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=RANDOM_STATE)
    elif CLUSTERING_ALGORITHM == 'spectral':
        return SpectralClustering(n_clusters=NUMBER_OF_CLUSTERS, random_state=RANDOM_STATE)

    raise Exception('Invalid CLUSTERING_ALGORITHM parameter')

def compute_cross_validation_spark_f(subset: pd.DataFrame, y: np.ndarray, q: Queue):
    """
    Computes a cross validations to get the concordance index in a Spark environment
    :param subset: Subset of features to compute the cross validation
    :param y: Y data
    :param q: Queue to return Process result
    """
    # Sets the default score (in case of error) to -1.0 if more_is_better (C-Index), otherwise 0.0
    error_score = -1.0 if MORE_IS_BETTER else 0.0

    try:
        n_features = subset.shape[1]

        # Locks to prevent multiple partitions in one worker getting all cores and degrading the performance
        logging.info(f'Waiting lock to compute CV with {n_features} features')
        with FileLock(f"/home/big_data/svm-surv.lock"):
            logging.info('File lock acquired, computing CV...')

            if MODEL_TO_USE == 'clustering':
                start = time.time()

                # Groups using the selected clustering algorithm
                clustering_model = get_clustering_model()
                clustering_result = clustering_model.fit(subset.values)

                # Generates a DataFrame with a column for time, event and the group
                labels = clustering_result.labels_
                dfs: List[pd.DataFrame] = []
                for cluster_id in range(NUMBER_OF_CLUSTERS):
                    current_group_y = y[np.where(labels == cluster_id)]
                    dfs.append(
                        pd.DataFrame({'E': current_group_y['event'], 'T': current_group_y['time'], 'group': cluster_id})
                    )
                df = pd.concat(dfs)

                # Fits a Cox Regression model using the column group as the variable to consider
                try:
                    cph = CoxPHFitter().fit(df, duration_col='T', event_col='E')

                    # This documentation recommends using log-likelihood to optimize:
                    # https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html#lifelines.fitters.coxph_fitter.SemiParametricPHFitter.score
                    fitness_value = cph.score(df, scoring_method=CLUSTERING_SCORING_METHOD)
                    end_time = time.time()
                    worker_execution_time = end_time - start  # Duplicated to not consider consumed time by all the metrics below
                except ConvergenceError:
                    fitness_value = -1
                    end_time = None
                    worker_execution_time = -1

                metric_description = 'Log Likelihood (higher is better)'
                mean_test_time = 0.0
                mean_train_score = 0.0
            else:
                logging.info(f'Computing CV ({NUMBER_OF_CV_FOLDS} folds) with {MODEL_TO_USE} model')
                start = time.time()
                cv_res = cross_validate(
                    CLASSIFIER,
                    subset,
                    y,
                    cv=NUMBER_OF_CV_FOLDS,
                    n_jobs=N_JOBS,
                    return_estimator=True,
                    return_train_score=RETURN_TRAIN_SCORE
                )
                fitness_value = cv_res['test_score'].mean()  # This is the C-Index
                end_time = time.time()
                worker_execution_time = end_time - start  # Duplicated to not consider consumed time by all the metrics below

                metric_description = 'Concordance Index (higher is better)'
                mean_test_time = np.mean(cv_res['score_time'])
                mean_train_score = cv_res['train_score'].mean() if RETURN_TRAIN_SCORE else 0.0

            logging.info(f'Fitness function with {n_features} features: {worker_execution_time} seconds | '
                         f'{metric_description}: {fitness_value}')

            partition_id = TaskContext().partitionId()

            # Gets a time-lapse description to check if some worker is lazy
            start_desc = datetime.fromtimestamp(start).strftime("%H:%M:%S")
            end_desc = datetime.fromtimestamp(end_time).strftime("%H:%M:%S") if end_time else '-'
            time_description = f'{start_desc} - {end_desc}'

            # 'res' is only defined when using SVM or RF

            # Gets number of iterations (only for SVM)
            if MODEL_TO_USE == 'svm':
                times_by_iteration = []
                total_number_of_iterations = []
                for estimator, fit_time in zip(cv_res['estimator'], cv_res['fit_time']):
                    # Scikit-surv doesn't use BaseLibSVM. So it doesn't have 'n_iter_' attribute
                    # number_of_iterations += np.sum(estimator.n_iter_)
                    number_of_iterations = estimator.optimizer_result_.nit
                    time_by_iterations = fit_time / number_of_iterations
                    times_by_iteration.append(time_by_iterations)
                    total_number_of_iterations.append(number_of_iterations)

                mean_times_by_iteration = np.mean(times_by_iteration)
                mean_total_number_of_iterations = np.mean(total_number_of_iterations)
            else:
                mean_times_by_iteration = 0.0
                mean_total_number_of_iterations = 0.0

            q.put([
                fitness_value,
                worker_execution_time,
                partition_id,
                socket.gethostname(),
                subset.shape[1],
                time_description,
                mean_times_by_iteration,
                mean_test_time,
                mean_total_number_of_iterations,
                mean_train_score
            ])
    except Exception as ex:
        logging.error('An exception has occurred in the fitness function:')
        logging.exception(ex)

        # Returns empty values
        q.put([
            error_score,  # Fitness value,
            -1.0,  # Worker time,
            -1.0,  # Partition ID,
            '',  # Host name,
            0,  # Number of features,
            '',  # Time description,
            -1.0,  # Mean times_by_iteration,
            -1.0,  # Mean test_time,
            -1.0,  # Mean total_number_of_iterations,
            error_score  # Mean train_score
        ])


def compute_cross_validation_spark(
        subset: Union[pd.DataFrame, Broadcast],
        index_array: np.ndarray,
        y: np.ndarray,
        is_broadcast: bool
) -> Tuple[float, float, int, str, int, str, float, float, float, float]:
    """
    Calls fitness inside a Process to prevent issues with memory leaks in Python.
    More info: https://stackoverflow.com/a/71700592/7058363
    :param is_broadcast: if True, the subset is a Broadcast instance
    :param index_array: Binary array where 1 indicates that the feature in that position must be included
    :param subset: Subset of features to compute the cross validation
    :param y: Y data
    :return: Result tuple with [0] -> fitness value, [1] -> execution time, [2] -> Partition ID, [3] -> Hostname,
    [4] -> number of evaluated features, [5] -> time-lapse description, [6] -> time by iteration and [7] -> avg test time
    [8] -> mean of number of iterations of the model inside the CV, [9] -> train score
    """
    # If broadcasting is enabled, the retrieves the Broadcast instance value
    x_values = subset.value if is_broadcast else subset

    q = Queue()
    parsed_data = get_columns_from_df(index_array, x_values)
    p = Process(target=compute_cross_validation_spark_f, args=(parsed_data, y, q))
    p.start()
    process_result = q.get()
    p.join()
    return process_result


def compute_cross_validation_sequential(df: pd.DataFrame, index_array: np.ndarray, y: np.ndarray,
                                        _is_broadcast: bool
                                        ) -> Tuple[float, float, None, None, int, str, float, float, float, float]:
    """
    Computes CV to get the Concordance Index
    :param df: Subset of features to be used in the model evaluated in the CrossValidation
    :param index_array: Binary array where 1 indicates that the feature in that position must be included
    :param y: Classes.
    :param _is_broadcast: Used for compatibility with Spark fitness function.
    :return: Average of the C-Index obtained in each CrossValidation fold
    """
    subset = get_columns_from_df(index_array, df)
    n_features = subset.shape[1]

    if MODEL_TO_USE == 'clustering':
        start = time.time()

        # Groups using the selected clustering algorithm
        clustering_model = get_clustering_model()
        clustering_result = clustering_model.fit(subset.values)

        # Generates a DataFrame with a column for time, event and the group
        labels = clustering_result.labels_
        dfs: List[pd.DataFrame] = []
        for cluster_id in range(NUMBER_OF_CLUSTERS):
            current_group_y = y[np.where(labels == cluster_id)]
            dfs.append(
                pd.DataFrame({'E': current_group_y['event'], 'T': current_group_y['time'], 'group': cluster_id})
            )
        df = pd.concat(dfs)

        # Fits a Cox Regression model using the column group as the variable to consider
        try:
            cph = CoxPHFitter().fit(df, duration_col='T', event_col='E')

            # This documentation recommends using log-likelihood to optimize:
            # https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html#lifelines.fitters.coxph_fitter.SemiParametricPHFitter.score
            fitness_value = cph.score(df, scoring_method=CLUSTERING_SCORING_METHOD)
            end_time = time.time()
            worker_execution_time = end_time - start  # Duplicated to not consider consumed time by all the metrics below
        except ConvergenceError:
            fitness_value = -1
            end_time = None
            worker_execution_time = -1

        metric_description = 'Log Likelihood (higher is better)'
        mean_test_time = 0.0
        mean_train_score = 0.0
    else:
        logging.info(f'Computing CV ({NUMBER_OF_CV_FOLDS} folds) with {MODEL_TO_USE} model')
        start = time.time()
        cv_res = cross_validate(
            CLASSIFIER,
            subset,
            y,
            cv=NUMBER_OF_CV_FOLDS,
            n_jobs=N_JOBS,
            return_estimator=True,
            return_train_score=RETURN_TRAIN_SCORE
        )
        fitness_value = cv_res['test_score'].mean()  # This is the C-Index
        end_time = time.time()
        worker_execution_time = end_time - start  # Duplicated to not consider consumed time by all the metrics below

        metric_description = 'Concordance Index (higher is better)'
        mean_test_time = np.mean(cv_res['score_time'])
        mean_train_score = cv_res['train_score'].mean() if RETURN_TRAIN_SCORE else 0.0

    logging.info(f'Fitness function with {n_features} features: {worker_execution_time} seconds | '
                 f'{metric_description}: {fitness_value}')

    # Gets a time-lapse description to check if some worker is lazy
    start_desc = datetime.fromtimestamp(start).strftime("%H:%M:%S")
    end_desc = datetime.fromtimestamp(end_time).strftime("%H:%M:%S") if end_time else '-'
    time_description = f'{start_desc} - {end_desc}'

    # 'res' is only defined when using SVM or RF

    # Gets number of iterations (only for SVM)
    if MODEL_TO_USE == 'svm':
        times_by_iteration = []
        total_number_of_iterations = []
        for estimator, fit_time in zip(cv_res['estimator'], cv_res['fit_time']):
            # Scikit-surv doesn't use BaseLibSVM. So it doesn't have 'n_iter_' attribute
            # number_of_iterations += np.sum(estimator.n_iter_)
            number_of_iterations = estimator.optimizer_result_.nit
            time_by_iterations = fit_time / number_of_iterations
            times_by_iteration.append(time_by_iterations)
            total_number_of_iterations.append(number_of_iterations)

        mean_times_by_iteration = np.mean(times_by_iteration)
        mean_total_number_of_iterations = np.mean(total_number_of_iterations)
    else:
        mean_times_by_iteration = 0.0
        mean_total_number_of_iterations = 0.0

    # This data is not used in the current version of the algorithm (sequential).
    # Adds them to the result to keep the same format as the Spark version
    partition_id = None
    host_name = None
    return (fitness_value, worker_execution_time, partition_id, host_name, n_features, time_description,
            mean_times_by_iteration, mean_test_time, mean_total_number_of_iterations, mean_train_score)


def main():
    # Generates parameters to train the ML model for load balancing and
    load_balancer_parameters = None
    if MODEL_TO_USE == 'svm':
        task = 'regression' if IS_SVM_REGRESSION else 'ranking'
        parameters_description = f'{task}_{MAX_ITERATIONS}_max_iterations_{SVM_OPTIMIZER}_optimizer_{SVM_KERNEL}_kernel'

        if USE_LOAD_BALANCER:
            load_balancer_parameters = SVMParameters(SVM_KERNEL, SVM_OPTIMIZER)
    else:
        if MODEL_TO_USE == 'rf':
            parameters_description = f'{RF_N_ESTIMATORS}_trees'
            if USE_LOAD_BALANCER:
                load_balancer_parameters = RFParameters(number_of_trees=RF_N_ESTIMATORS)
        else:
            clustering_scoring_method = CLUSTERING_SCORING_METHOD.replace('_', '-')
            parameters_description = (f'{NUMBER_OF_CLUSTERS}_clusters_{CLUSTERING_ALGORITHM}_'
                                      f'algorithm_{clustering_scoring_method}_score_method')
            if USE_LOAD_BALANCER:
                load_balancer_parameters = ClusteringParameters(algorithm=CLUSTERING_ALGORITHM,
                                                                number_of_clusters=NUMBER_OF_CLUSTERS,
                                                                scoring_method=CLUSTERING_SCORING_METHOD)

    # Spark settings
    app_name = f"BBHA_{time.time()}".replace('.', '_')
    if RUN_IN_SPARK:
        conf = SparkConf().setMaster(MASTER_CONNECTION_URL).setAppName(app_name)

        if EXECUTORS is not None:
            conf = conf.set("spark.executor.instances", EXECUTORS)

        if CORES_PER_EXECUTOR is not None:
            conf = conf.set("spark.executor.cores", CORES_PER_EXECUTOR)

        if MEMORY_PER_EXECUTOR is not None:
            conf = conf.set("spark.executor.memory", MEMORY_PER_EXECUTOR)

        # Gets Spark context
        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        sc = spark.sparkContext
        sc.setLogLevel("ERROR")

        # Gets the number of workers
        spark_2 = SparkContext.getOrCreate(conf=conf)
        sc2 = spark_2._jsc.sc()
        number_of_workers = len([executor.host() for executor in
                                 sc2.statusTracker().getExecutorInfos()]) - 1  # Subtract 1 to discard the master

        fitness_function = compute_cross_validation_spark
    else:
        fitness_function = compute_cross_validation_sequential
        sc = None
        number_of_workers = 0  # No Spark, no workers

    run_improved_bbha = False if RUN_IN_SPARK else RUN_IMPROVED_BBHA  # TODO: improved BBHA it's not implemented for Spark right now
    add_epsilon = MODEL_TO_USE == 'svm'

    if RUN_TIMES_EXPERIMENT:
        # Runs an experiment to store execution times, models parameters and other data to train a load balancer
        if RUN_IN_SPARK:
            run_times_experiment(
                app_name=app_name,
                compute_cross_validation=compute_cross_validation_spark,
                n_iterations=N_ITERATIONS,
                number_of_independent_runs=NUMBER_OF_INDEPENDENT_RUNS,
                n_stars=N_STARS,
                load_balancer_parameters=load_balancer_parameters,
                sc=sc,
                metric_description='concordance index',
                add_epsilon=add_epsilon,  # Epsilon is needed only by the SVM
                dataset=DATASET,
                number_of_workers=number_of_workers,
                model_name=MODEL_TO_USE,
                parameters_description=parameters_description,
                use_broadcasts_in_spark=USE_BROADCAST,
                debug=DEBUG
            )
        else:
            # Runs sequentially
            run_times_experiment_sequential(
                compute_cross_validation=compute_cross_validation_sequential,
                n_iterations=NUMBER_OF_INDEPENDENT_RUNS,
                metric_description='concordance index',
                add_epsilon=add_epsilon,  # Epsilon is needed only by the SVM
                model_name=MODEL_TO_USE,
                parameters_description=parameters_description,
            )
    else:
        # For clustering there are two metrics: log-likelihood and concordance-index
        if MODEL_TO_USE == 'clustering':
            metric_description = CLUSTERING_SCORING_METHOD.replace('_', '-')
        else:
            # For the rest of the models (SVM/RF) there's only one metric: concordance-index
            metric_description = 'concordance-index'

        # Runs normal Feature Selection experiment
        run_experiment(
            app_name=app_name,
            more_is_better=MORE_IS_BETTER,
            run_improved_bbha=run_improved_bbha,
            run_in_spark=RUN_IN_SPARK,
            n_stars=N_STARS,
            random_state=RANDOM_STATE,
            compute_cross_validation=fitness_function,
            sc=sc,
            metric_description=metric_description,
            add_epsilon=add_epsilon,  # Epsilon is needed only by the SVM
            debug=DEBUG,
            dataset=DATASET,
            load_balancer_parameters=load_balancer_parameters,
            number_of_independent_runs=NUMBER_OF_INDEPENDENT_RUNS,
            n_iterations=N_ITERATIONS,
            number_of_workers=number_of_workers,
            model_name=MODEL_TO_USE,
            parameters_description=parameters_description,
            use_broadcasts_in_spark=USE_BROADCAST
        )


if __name__ == '__main__':
    main()
