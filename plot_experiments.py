import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict, Literal, cast, Tuple, Optional
import pandas as pd
from sklearn.metrics import mean_squared_error

ModelKey = Literal['best_gradient_booster_model', 'best_gradient_booster_model_no_min_max', 'best_linear_model_3',
'best_linear_model_3_no_min_max', 'best_nn_model', 'best_nn_model_no_min_max']

# RESULTS_FOLDER_PATH = './Results'
RESULTS_FOLDER_PATH = 'paper_load_balancing'

# Idle/Execution times structure for bar charts. Keys are workers, values is a List of lists with the iteration number
# and the time.
WorkerBarTimes = Dict[str, List[List[float]]]

DataType = Literal['Idle', 'Execution']

# Number of stars to divide the sequential results
NUMBER_OF_STARS = 60

# Folder name to save the images (this will be inside the RESULTS_FOLDER_PATH folder)
SAVE_FOLDER_IMGS_NAME = 'images'

# If True, saves the images in the RESULTS_FOLDER_PATH/SAVE_FOLDER_IMGS_NAME folder
SAVE_IMGS = False
# SAVE_IMGS = True

# Folder name to save the images (this will be inside the RESULTS_FOLDER_PATH folder)
SAVE_FOLDER_CSV_NAME = 'csvs'

# If True, saves the CSV files in the RESULTS_FOLDER_PATH/SAVE_FOLDER_CSV_NAME folder
SAVE_CSV_FILES = True

# If True, shows the images
# PLOT_IMAGES = True
PLOT_IMAGES = False


def __save_img(title: str):
    """
    Saves the current img in the RESULTS_FOLDER_PATH/SAVE_FOLDER_IMGS_NAME folder (if SAVE_IMGS is True).
    :param title: Title of the image
    """
    if SAVE_IMGS:
        save_path = os.path.join(RESULTS_FOLDER_PATH, SAVE_FOLDER_IMGS_NAME)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Replaces the \n with _, and reduces the title to 90 characters to prevent issues in Windows
        fig_path = os.path.join(save_path, f'{title}').replace('\n', '_')[:90].strip()
        print(f'"{fig_path}.png"')
        plt.savefig(fig_path + '.png')


def __save_csv(csv_data: dict, title: str):
    """
    Saves the CSV data in the RESULTS_FOLDER_PATH/SAVE_FOLDER_CSV_NAME folder (if SAVE_CSV_FILES is True).
    :param title: Title of the image
    """
    if SAVE_CSV_FILES:
        save_path = os.path.join(RESULTS_FOLDER_PATH, SAVE_FOLDER_CSV_NAME)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Replaces the \n with _
        fig_path = os.path.join(save_path, f'{title}').replace('\n', '_')[:90].strip()
        df = pd.DataFrame(csv_data)
        df.to_csv(fig_path + '.csv', index=False)


def generate_bar_charts(data: WorkerBarTimes, title: str, data_type: DataType):
    """
    Plots a bar chart
    :param data: Dictionary with the worker name and the number of id
    :param title: Title to show in the bar chart
    :param data_type: To show 'Idle' or 'Execution' in bar chart title.
    """
    _fig, ax = plt.subplots()

    # Adds some text for labels, title and axes ticks
    ax.set_ylabel(f'{data_type} time (seconds)')
    fig_title = f'{data_type[:4]} {title}'  # Makes the title shorter
    ax.set_title(fig_title)

    # In case SAVE_CSV_FILES is True, creates a dict with the data to save it in a CSV file
    csv_data: Optional[Dict[str, List[float]]]
    csv_summary_data: Optional[Dict[str, List[float]]]
    if SAVE_CSV_FILES:
        csv_data = {
            'Iteration': [],
            'Worker': [],
            'Time': []
        }
    else:
        csv_data = None

    width = 0.25
    iterations: np.ndarray = np.array([])  # Just to prevent MyPy warning
    for idx, worker in enumerate(data.keys()):
        np_array = np.array(data[worker])
        iterations = np_array[:, 0] + 1  # +1 to start from 1 instead of 0
        data_times_per_iteration = np_array[:, 1]
        # print(f'Worker {worker} | {data_type} times: {data_times_per_iteration}')
        margin = width * idx
        plt.bar(iterations + margin, data_times_per_iteration, width=width, label=worker)

        if SAVE_CSV_FILES:
            csv_data['Iteration'].extend(iterations)
            csv_data['Worker'].extend([worker] * len(iterations))
            csv_data['Time'].extend(data_times_per_iteration)

    if SAVE_CSV_FILES:
        # Makes a summary making a mean and std of the times per iteration to check if there's some improvement.
        # Takes the data from the csv_data dict
        csv_summary_data = {
            'Iteration': [],
            'Mean': [],
            'Std': []
        }

        # Iterates over iterations for the current worker
        for iteration in np.unique(iterations):
            idxs = np.array(csv_data['Iteration']) == iteration
            times = np.array(csv_data['Time'])[idxs]

            # Gets the mean and std
            mean = np.mean(times)
            std = np.std(times)

            # Appends to the summary dict
            csv_summary_data['Iteration'].append(iteration)
            csv_summary_data['Mean'].append(mean)
            csv_summary_data['Std'].append(std)
    else:
        csv_summary_data = None

    plt.xticks(iterations)
    plt.legend()

    # Saves images and CSV files
    __save_img(fig_title)
    __save_csv(csv_data, fig_title)
    __save_csv(csv_summary_data, f'{fig_title} summary')


def generate_predictions_line_charts(n_features: List[float], execution_times: List[float],
                                     predicted_times: List[float], title: str):
    """
    Plots a line chart with the execution time and the predicted time.
    :param n_features: List with number of features.
    :param execution_times: List with execution times. Have to have the same order as n_features.
    :param predicted_times: List with predicted execution times. Have to have the same order as n_features.
    :param title: Title to show in the Line chart's title.
    """
    _fig, ax = plt.subplots()

    # Sorts by number of features ascending preserving the order in all the lists
    n_features = np.array(n_features)
    execution_times = np.array(execution_times)
    predicted_times = np.array(predicted_times)

    sorted_idx = n_features.argsort()
    n_features = n_features[sorted_idx[::]]
    execution_times = execution_times[sorted_idx[::]]
    predicted_times = predicted_times[sorted_idx[::]]

    execution_means = []
    execution_std_errors = []
    predicted_means = []
    predicted_std_errors = []

    if SAVE_CSV_FILES:
        csv_data: Dict[str, List[float]] | None = {
            'Number of features': [],
            'Execution time': [],
            'Predicted time': [],
            'Mean squared error': []
        }
    else:
        csv_data = None

    unique_n_features = np.unique(n_features)
    for current_n_features in unique_n_features:
        # Gets index of the current number of features to get the execution/predicted times
        idx = np.where(n_features == current_n_features)

        # Stores execution times
        current_execution_times = execution_times[idx]
        execution_mean, std_error = get_mean_and_std(current_execution_times)

        execution_means.append(execution_mean)
        execution_std_errors.append(std_error)

        # Stores predictions times
        current_predicted_times = predicted_times[idx]
        predicted_mean, std_error = get_mean_and_std(current_predicted_times)

        # The predictions shouldn't change
        try:
            assert round(std_error, 10) == 0  # Sometimes it could be tiny (1e-16)
        except AssertionError as ex:
            print(f'The experiment with title "{title}" has predictions with std != 0. Maybe it\'s an old experiment'
                  f'where the predictions were stored in an incorrect order.')
            raise ex

        predicted_means.append(predicted_mean)
        predicted_std_errors.append(std_error)

    # Plots execution and predicted times
    ax.errorbar(unique_n_features, execution_means, yerr=execution_std_errors, capsize=4, label='Execution time',
                marker='o', linewidth=2)
    ax.errorbar(unique_n_features, predicted_means, yerr=predicted_std_errors, capsize=4, label='Predicted time',
                marker='o', linewidth=2)

    plt.xlabel('Number of features')
    plt.ylabel(f'Time (seconds)')
    plt.legend(loc='upper right')
    fig_title = f'Execution and predicted times. {title}'
    plt.title(f'Execution and predicted times. {title}')

    if SAVE_CSV_FILES:
        csv_data['Number of features'].extend(unique_n_features)
        csv_data['Execution time'].extend(execution_means)
        csv_data['Predicted time'].extend(predicted_means)
        csv_data['Mean squared error'].extend([(execution_mean - predicted_mean) ** 2 for execution_mean,
                                               predicted_mean in zip(execution_means, predicted_means)])

    # Saves images and CSV files
    __save_img(fig_title)
    __save_csv(csv_data, fig_title)


def get_mean_and_std(values: np.ndarray) -> Tuple[float, float]:
    """Gets mean and std of a list of floats"""
    execution_mean: float = cast(float, np.mean(values))
    if len(values) > 1:
        std_error = np.std(values, ddof=1) / np.sqrt(len(values))
    else:
        std_error = 0.0
    return execution_mean, std_error


class Experiment:
    """Represent a Result of an experiment"""
    dir_path: str
    name: str
    plot_predictions: bool
    all_models_predictions_file: str
    original_model_key: ModelKey
    n_stars: int
    # If True, only considers data that are present in the iterations where all the workers have participated.
    only_consider_intersection: bool
    # If True gets the key 'execution_times' instead of the 'workers_idle_times_per_iteration' or
    # 'workers_execution_times_per_iteration'
    is_sequential: bool

    def __init__(self, file_path: str, name: str, plot_predictions: bool, model_name_in_json: str = None,
                 original_model_key: ModelKey = 'best_gradient_booster_model', n_stars=90,
                 only_consider_intersection: bool = False, is_sequential=False):
        self.dir_path = file_path
        self.name = name
        self.plot_predictions = plot_predictions
        self.all_models_predictions_file = model_name_in_json
        self.original_model_key = original_model_key
        self.n_stars = n_stars
        self.only_consider_intersection = only_consider_intersection
        self.is_sequential = is_sequential


def __sum_data_by_iteration(original_dict: dict, current_dict: dict):
    """
    Sums the time to the corresponding iteration in the original dict. All of this by every worker.
    :param original_dict: Original dict to sum the times.
    :param current_dict: Dict with the times to sum.
    """
    for worker in current_dict.keys():
        for iteration, execution_time in current_dict[worker]:
            for elem in original_dict[worker]:
                if iteration == elem[0]:
                    elem[1] += execution_time
                    break


def get_times_data(experiment: Experiment,
                   data_key: Literal['workers_idle_times_per_iteration', 'workers_execution_times_per_iteration',
                   'predicted_execution_times']) -> WorkerBarTimes:
    """
    Gets bar and line charts data
    :param experiment: Experiment instance
    :param data_key: Data to retrieve
    :return: Dicts ready to use as BarChart or LineChart
    """
    bar_chart_data_aux: Optional[WorkerBarTimes] = None

    # Iterates over all JSON files in the Experiment's folder
    dir_path = os.path.join(RESULTS_FOLDER_PATH, experiment.dir_path)

    # If not exists raise an exception
    if not os.path.exists(dir_path):
        raise Exception(f'Experiment with name "{experiment.name}" and dir path "{dir_path}" does not exist.')

    for (_, _, files) in os.walk(dir_path):
        for filename in files:
            if not filename.endswith('.json'):
                continue

            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r') as fp:
                json_content: Dict = json.load(fp)

            json_data = json_content[data_key]

            # Appends all the idle times for all the iterations
            if data_key == 'execution_times':

                # Filters first NUMBER_OF_STARS elements (initialization of stars which are not considered in the
                # Spark experiments) and converts to numpy array
                json_data = np.array(json_data[NUMBER_OF_STARS:])

                # Divides the data into NUMBER_OF_STARS splits
                number_of_partitions = int(len(json_data) / NUMBER_OF_STARS)
                json_data = np.array_split(json_data, number_of_partitions)

                # In case of sequential experiment, simulates the same structure as Spark experiment
                sequential_json_data = {'Master': []}
                for current_iteration, values in enumerate(json_data):
                    sequential_json_data['Master'].append([current_iteration, np.sum(values)])

                # Sets to json_data to continue with the normal algorithm
                json_data = sequential_json_data

            if bar_chart_data_aux is None:
                # TODO: implement filter by in common iterations
                bar_chart_data_aux = json_data
            else:
                __sum_data_by_iteration(bar_chart_data_aux, json_data)

    # Filters by common intersection
    if experiment.only_consider_intersection:
        bar_chart_data: WorkerBarTimes = {}

        # Gets iterations in common
        intersection = None
        all_iterations = []
        for host_name, values in bar_chart_data_aux.items():
            iterations = np.array(values)[:, 0]
            if intersection is not None:
                intersection = np.intersect1d(intersection, iterations)
            else:
                intersection = iterations

            # Adds iterations to all iterations
            all_iterations.extend(iterations)

        # Reports non intersection
        non_intersection = np.setdiff1d(all_iterations, intersection).astype(int)
        if len(non_intersection) > 0:
            print(
                f'Experiment "{experiment.name}" has non intersection iterations: {", ".join(non_intersection.astype(str))}')

        for worker, values in bar_chart_data_aux.items():
            bar_chart_data[worker] = []

            for iteration_number, time in values:
                if iteration_number in intersection:
                    bar_chart_data[worker].append([iteration_number, time])
    else:
        bar_chart_data = bar_chart_data_aux

    return bar_chart_data


def get_predictions_data(experiment: Experiment) -> Tuple[List[float], List[float], List[float]]:
    """
    Gets the number of features, execution times and predicted times (preserving original order for all the three
    lists). Ready to use in the Line chart of predicted times.
    :param experiment: Experiment instance to get.
    :return: Three list with the ready-to-use data.
    """
    number_of_features: List[float] = []
    execution_times: List[float] = []
    predicted_execution_times: List[float] = []

    # Iterates over all JSON files in the Experiment's folder
    dir_path = os.path.join(RESULTS_FOLDER_PATH, experiment.dir_path)
    for (_, _, files) in os.walk(dir_path):
        for filename in files:
            if not filename.endswith('.json'):
                continue

            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r') as fp:
                json_content: Dict = json.load(fp)

            # Appends all the data
            number_of_features.extend(json_content['number_of_features'])
            execution_times.extend(json_content['execution_times'])
            predicted_execution_times.extend(json_content['predicted_execution_times'])

    print(f'MSE of {experiment.name}: {mean_squared_error(execution_times, predicted_execution_times)}')

    return number_of_features, execution_times, predicted_execution_times


def plot_charts(experiment: Experiment,
                data_key: Literal['workers_idle_times_per_iteration', 'workers_execution_times_per_iteration',
                'execution_times']):
    """Shows idle/execution times for an experiment in a bar and line chart."""
    bar_chart_data = get_times_data(experiment, data_key)

    is_plotting_idle_times = data_key == 'workers_idle_times_per_iteration'
    data_type: DataType
    if is_plotting_idle_times:
        data_type = 'Idle'
    else:
        data_type = 'Execution'

    common_it_str = ' (common iterations)' if experiment.only_consider_intersection else ''
    title = f'{experiment.name}{common_it_str}'
    generate_bar_charts(bar_chart_data, title=title, data_type=data_type)

    # If needed plots predictions times
    if is_plotting_idle_times and experiment.plot_predictions:
        n_features, execution_times, predicted_times = get_predictions_data(experiment)
        generate_predictions_line_charts(n_features, execution_times, predicted_times, title=experiment.name)


def main():
    experiments: List[Experiment] = [
        #     Experiment('breast_with_load_balancer', 'Breast (with G.B. load balancer)', plot_predictions=True),
        #     Experiment('breast_with_load_balancer', 'Breast (with G.B. load balancer)',
        #                plot_predictions=True),
        #     Experiment('lung_with_load_balancer', 'Lung (with G.B. load balancer)', plot_predictions=True),
        #                model_name_in_json='/home/genaro/logs_30_ind_it_30_it_90_stars_lung_with_load_balancer.txt'),
        #     Experiment('lung_with_load_balancer_better_gradient_booster', 'Lung (with G.B. LB and better model)',
        #                plot_predictions=True, model_name_in_json='/home/genaro/logs_30_ind_it_30_it_90_stars_lung_with_load_balancer_try_2_with_better_gradient_booster_model.txt'),
        #     Experiment('lung_with_nn_as_load_balancer', 'Lung (with NN LB and better model)',
        #                plot_predictions=True), #, model_name_in_json='/home/genaro/logs_30_ind_it_30_it_90_stars_lung_with_load_balancer_try_3_with_nn_model.txt', original_model_key='best_nn_model'),
        #     Experiment('lung_with_gb_as_load_balancer_trained_with_lung', 'Lung (with G.B. LB and better model trained with lung data)',
        #                plot_predictions=True), #, model_name_in_json='/home/genaro/logs_30_ind_it_30_it_90_stars_lung_with_load_balancer_try_4_with_gb_model_trained_with_lung.txt'),
        #     Experiment('lung_with_gb_as_load_balancer_trained_with_logs_fixed_predictions',
        #                'Lung fixed predictions (with G.B. LB and better model trained with lung data)',
        #                plot_predictions=True, only_consider_intersection=False), # model_name_in_json='/home/genaro/logs_30_ind_it_30_it_90_stars_lung_with_load_balancer_try_6_with_gb_model_trained_with_lung_more_fixed_predictions.txt')

        # AFTER FIXING PREDICTIONS
        # Experiment('breast_without_load_balancer', 'Breast (without load balancer)', plot_predictions=False, only_consider_intersection=True),
        # Experiment('breast_with_load_balancer_fixed_predictions', 'Breast fixed predictions (G.B. LB first model\n Trained model "svm_first_attempt")',
        #            plot_predictions=True, only_consider_intersection=True),
        # Experiment('lung_without_load_balancer', 'Lung (without load balancer)', plot_predictions=False, only_consider_intersection=True),
        # Experiment('lung_with_gb_as_load_balancer_trained_with_logs_fixed_predictions',
        #            'Lung fixed predictions (G.B. load balancer, better model trained with lung data\nTrained model "svm")',
        #            plot_predictions=True, only_consider_intersection=True)

        # WITH ONLY TWO WORKERS (FINAL)
        # Experiment('breast_without_load_balancer_2_workers_final', 'Breast 2 workers (without load balancer)', plot_predictions=False,
        #            only_consider_intersection=True),
        # Experiment('breast_with_load_balancer_2_workers_final', 'Breast 2 workers (with load balancer)', plot_predictions=True,
        #            only_consider_intersection=True),
        # Experiment('lung_without_load_balancer_2_workers_final', 'Lung 2 workers (without load balancer)', plot_predictions=False,
        #            only_consider_intersection=True),
        # Experiment('lung_with_load_balancer_2_workers_final', 'Lung 2 workers (with load balancer)', plot_predictions=True,
        #            only_consider_intersection=True),

        # WITH ONLY TWO WORKERS TIMES (FINAL)
        # Experiment('breast_without_load_balancer_2_workers_final_times', 'Breast 2 workers times (without load balancer)', plot_predictions=False,
        #            only_consider_intersection=True),
        # Experiment('breast_with_load_balancer_2nd_model_2_workers_final_times', 'Breast 2 workers times (with 2nd model load balancer)', plot_predictions=True,
        #            only_consider_intersection=True),
        # Experiment('breast_with_load_balancer_3rd_model_2_workers_final_times', 'Breast 2 workers times (with 3rd load balancer)', plot_predictions=True,
        #            only_consider_intersection=True),
        # Experiment('lung_without_load_balancer_2_workers_final_times', 'Lung 2 workers times (without load balancer)', plot_predictions=False,
        #            only_consider_intersection=True),
        # Experiment('lung_with_load_balancer_2nd_model_2_workers_final_times', 'Lung 2 workers times (with 2nd load balancer)', plot_predictions=True,
        #            only_consider_intersection=True),
        # Experiment('lung_with_load_balancer_3rd_model_2_workers_final_times', 'Lung 2 workers times (with 3rd load balancer)', plot_predictions=True,
        #            only_consider_intersection=True),

        # Paper load balancer. NOTE: RESULTS_FOLDER_PATH = './paper_load_balancing'

        # Breast, Clustering, K-means, 2 clusters
        Experiment('clustering/breast/logs_breast_clustering_sequential_k_means_log_likelihood_2_clusters_1_core',
                   'Breast Sequential K-means',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('clustering/breast/logs_breast_clustering_n_stars_k_means_log_likelihood_2_clusters_1_core',
                   'Breast N-Stars K-means',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('clustering/breast/logs_breast_clustering_binpacking_k_means_log_likelihood_2_clusters_1_core',
                   'Breast LB (v1) K-means',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('clustering/breast'
                   '/logs_breast_clustering_new_binpacking_model_k_means_log_likelihood_2_clusters_1_core',
                   'Breast LB (full) K-means',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment(
            'clustering/breast/logs_breast_clustering_overfitting_binpacking_model_k_means_log_likelihood_2_clusters_1_core',
            'Breast LB (overfitting) K-means',
            plot_predictions=True, only_consider_intersection=True),

        # Breast, SVM, Linear
        Experiment('svm/breast/logs_breast_svm_sequential_kernel_linear_regression_1_core_6g',
                   'Breast Sequential SVM Lin',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/breast/logs_breast_svm_n_stars_kernel_linear_regression_1_core', 'Breast N-Stars SVM Lin',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_binpacking_kernel_linear_regression_1_core',
                   'Breast LB (v1) SVM Lin',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_new_binpacking_model_kernel_linear_regression_1_core',
                   'Breast LB (full) SVM Lin',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_overfitting_binpacking_model_kernel_linear_regression_1_core',
                   'Breast LB (overfitting) SVM Lin',
                   plot_predictions=True, only_consider_intersection=True),

        # Breast, SVM, Poly
        Experiment('svm/breast/logs_breast_svm_sequential_kernel_poly_regression_1_core_6g',
                   'Breast Sequential SVM Poly',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/breast/logs_breast_svm_n_stars_kernel_poly_regression_1_core', 'Breast N-Stars SVM Poly',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_binpacking_kernel_poly_regression_1_core',
                   'Breast LB (v1) SVM Poly',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_new_binpacking_model_kernel_poly_regression_1_core',
                   'Breast LB (full) SVM Poly',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_overfitting_binpacking_model_kernel_poly_regression_1_core',
                   'Breast LB (overfitting) SVM Poly',
                   plot_predictions=True, only_consider_intersection=True),

        # Breast, SVM, RBF
        Experiment('svm/breast/logs_breast_svm_sequential_kernel_rbf_regression_1_core_6g',
                   'Breast Sequential SVM RBF',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/breast/logs_breast_svm_n_stars_kernel_rbf_regression_1_core', 'Breast N-Stars SVM RBF',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_binpacking_kernel_rbf_regression_1_core',
                   'Breast LB (v1) SVM RBF',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_new_binpacking_model_kernel_rbf_regression_1_core',
                   'Breast LB (full) SVM RBF',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_overfitting_binpacking_model_kernel_rbf_regression_1_core',
                   'Breast LB (overfitting) SVM RBF',
                   plot_predictions=True, only_consider_intersection=True),

        # Kidney, Clustering, K-means, 2 clusters
        Experiment('clustering/kidney/logs_kidney_clustering_sequential_k_means_log_likelihood_2_clusters_1_core_6g',
                   'Kidney Sequential K-means',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('clustering/kidney/logs_kidney_clustering_n_stars_k_means_log_likelihood_2_clusters_1_core',
                   'Kidney N-Stars K-means',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('clustering/kidney/logs_kidney_clustering_binpacking_k_means_log_likelihood_2_clusters_1_core',
                   'Kidney LB (v1) K-means',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment(
            'clustering/kidney/logs_kidney_clustering_new_binpacking_model_k_means_log_likelihood_2_clusters_1_core',
            'Kidney LB (full) K-means',
            plot_predictions=True, only_consider_intersection=True),
        Experiment(
            'clustering/kidney/logs_kidney_clustering_overfitting_binpacking_model_k_means_log_likelihood_2_clusters_1_core',
            'Kidney LB (overfitting) K-means',
            plot_predictions=True, only_consider_intersection=True),

        # Kidney, SVM, Linear
        Experiment('svm/kidney/logs_kidney_svm_sequential_kernel_linear_regression_1_core_6g',
                   'Kidney Sequential SVM Lin',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/kidney/logs_kidney_svm_n_stars_kernel_linear_regression_1_core', 'Kidney N-Stars SVM Lin',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_binpacking_kernel_linear_regression_1_core',
                   'Kidney LB (v1) SVM Lin',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_new_binpacking_model_kernel_linear_regression_1_core',
                   'Kidney LB (full) SVM Lin',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_overfitting_binpacking_model_kernel_linear_regression_1_core',
                   'Kidney LB (overfitting) SVM Lin',
                   plot_predictions=True, only_consider_intersection=True),

        # Kidney, SVM, Poly
        Experiment('svm/kidney/logs_kidney_svm_sequential_kernel_poly_regression_1_core_6g',
                   'Kidney Sequential SVM Poly',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/kidney/logs_kidney_svm_n_stars_kernel_poly_regression_1_core', 'Kidney N-Stars SVM Poly',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_binpacking_kernel_poly_regression_1_core',
                   'Kidney LB (v1) SVM Poly',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_new_binpacking_model_kernel_poly_regression_1_core',
                   'Kidney LB (full) SVM Poly',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_overfitting_binpacking_model_kernel_poly_regression_1_core',
                   'Kidney LB (overfitting) SVM Poly',
                   plot_predictions=True, only_consider_intersection=True),

        # Kidney, SVM, RBF
        Experiment('svm/kidney/logs_kidney_svm_sequential_kernel_rbf_regression_1_core_6g',
                   'Kidney Sequential SVM RBF',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/kidney/logs_kidney_svm_n_stars_kernel_rbf_regression_1_core', 'Kidney N-Stars SVM RBF',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_binpacking_kernel_rbf_regression_1_core',
                   'Kidney LB (v1) SVM RBF',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_new_binpacking_model_kernel_rbf_regression_1_core',
                   'Kidney LB (full) SVM RBF',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_overfitting_binpacking_model_kernel_rbf_regression_1_core',
                   'Kidney LB (overfitting) SVM RBF',
                   plot_predictions=True, only_consider_intersection=True),
    ]

    for experiment in experiments:
        # Plots idle and execution times in bar charts
        if not experiment.is_sequential:
            plot_charts(experiment, data_key='workers_idle_times_per_iteration')

        execution_times_key = 'execution_times' if experiment.is_sequential else 'workers_execution_times_per_iteration'
        plot_charts(experiment, data_key=execution_times_key)

    # Shows all the charts
    if PLOT_IMAGES:
        plt.show()


if __name__ == '__main__':
    if not PLOT_IMAGES and not SAVE_IMGS and not SAVE_CSV_FILES:
        raise Exception('PLOT_IMAGES, SAVE_IMGS and SAVE_CSV_FILES cannot be both False.')
    main()
