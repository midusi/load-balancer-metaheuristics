import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict, Literal, cast, Tuple, Optional
from sklearn.metrics import mean_squared_error


ModelKey = Literal['best_gradient_booster_model', 'best_gradient_booster_model_no_min_max', 'best_linear_model_3',
                     'best_linear_model_3_no_min_max', 'best_nn_model', 'best_nn_model_no_min_max']

# RESULTS_FOLDER_PATH = './Results'
RESULTS_FOLDER_PATH = './paper_load_balancing'

# Idle/Execution times structure for bar charts. Keys are workers, values is a List of lists with the iteration number
# and the time.
WorkerBarTimes = Dict[str, List[List[float]]]

DataType = Literal['Idle', 'Execution']

# Number of stars to divide the sequential results
NUMBER_OF_STARS = 60

# Folder name to save the images (this will be inside the RESULTS_FOLDER_PATH folder)
SAVE_FOLDER_NAME = 'images'

# If True, saves the images in the RESULTS_FOLDER_PATH/SAVE_FOLDER_NAME folder
# SAVE_IMGS = False
SAVE_IMGS = True

# If True, shows the images
# PLOT_IMAGES = True
PLOT_IMAGES = False

def __save_img(title: str):
    """
    Saves the current img in the RESULTS_FOLDER_PATH/SAVE_FOLDER_NAME folder (if SAVE_IMGS is True).
    :param title: Title of the image
    """
    if SAVE_IMGS:
        save_path = os.path.join(RESULTS_FOLDER_PATH, SAVE_FOLDER_NAME)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{title}.png'))


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

    plt.xticks(iterations)
    plt.legend()

    __save_img(fig_title)


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
    ax.errorbar(unique_n_features, execution_means, yerr=execution_std_errors, capsize=4, label='Execution time', marker='o', linewidth=2)
    ax.errorbar(unique_n_features, predicted_means, yerr=predicted_std_errors, capsize=4, label='Predicted time', marker='o', linewidth=2)

    plt.xlabel('Number of features')
    plt.ylabel(f'Time (seconds)')
    plt.legend(loc='upper right')
    fig_title = f'Execution and predicted times. {title}'
    plt.title(f'Execution and predicted times. {title}')

    __save_img(fig_title)

def get_mean_and_std(values: np.ndarray) -> Tuple[float, float]:
    """Gets mean and std of a list of floats"""
    execution_mean: float = cast(float, np.mean(values))
    if len(values) > 1:
        std_error = np.std(values, ddof=1) / np.sqrt(len(values))
    else:
        std_error = 0.0
    return execution_mean, std_error


def generate_line_charts(data: Dict[str, List[float]], title: str, data_type: DataType):
    """
    Plots a line chart with all the workers and their idle/execution times.
    :param data: Dict with workers as keys and a list of Idle/Execution times as values.
    :param title: Title to show as the chart's title.
    :param data_type: 'Idle'/'Execution' to show in chart's title.
    """
    _fig, ax = plt.subplots()
    for worker, values in data.items():
        values_np = np.array(values)
        unique_iterations = np.unique(values_np[:, 0])
        means = []
        std_errors = []

        for it in unique_iterations:
            idx_it = np.where(values_np[:, 0] == it)
            times = values_np[idx_it][:, 1]

            # Gets mean and error
            mean = np.mean(times)
            means.append(mean)
            std_error = np.std(times, ddof=1) / np.sqrt(len(times))
            std_errors.append(std_error)

        # Plots idle/execution times for the current worker
        iterations = unique_iterations + 1
        ax.errorbar(iterations, means, yerr=std_errors, capsize=4, label=worker, marker='o', linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel(f'{data_type} time (seconds)')
    plt.legend(loc='upper right')
    fig_title = f'{data_type} time per worker. {title}'
    plt.title(fig_title)

    __save_img(fig_title)

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
                 original_model_key: ModelKey = 'best_gradient_booster_model', n_stars = 90,
                 only_consider_intersection: bool = False, is_sequential = False):
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
                line_chart_data_aux = json_data
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
            print(f'Experiment "{experiment.name}" has non intersection iterations: {", ".join(non_intersection.astype(str))}')

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

    title = f'{experiment.name}\nOnly common iterations: {experiment.only_consider_intersection}'
    generate_bar_charts(bar_chart_data, title=title, data_type=data_type)
    # generate_line_charts(bar_chart_data, title=title, data_type=data_type)  # FIXME: check if needed

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
    #     Experiment('lung_with_load_balancer_better_gradient_booster', 'Lung (with G.B. load balancer and better model)',
    #                plot_predictions=True, model_name_in_json='/home/genaro/logs_30_ind_it_30_it_90_stars_lung_with_load_balancer_try_2_with_better_gradient_booster_model.txt'),
    #     Experiment('lung_with_nn_as_load_balancer', 'Lung (with NN load balancer and better model)',
    #                plot_predictions=True), #, model_name_in_json='/home/genaro/logs_30_ind_it_30_it_90_stars_lung_with_load_balancer_try_3_with_nn_model.txt', original_model_key='best_nn_model'),
    #     Experiment('lung_with_gb_as_load_balancer_trained_with_lung', 'Lung (with G.B. load balancer and better model trained with lung data)',
    #                plot_predictions=True), #, model_name_in_json='/home/genaro/logs_30_ind_it_30_it_90_stars_lung_with_load_balancer_try_4_with_gb_model_trained_with_lung.txt'),
    #     Experiment('lung_with_gb_as_load_balancer_trained_with_logs_fixed_predictions',
    #                'Lung fixed predictions (with G.B. load balancer and better model trained with lung data)',
    #                plot_predictions=True, only_consider_intersection=False), # model_name_in_json='/home/genaro/logs_30_ind_it_30_it_90_stars_lung_with_load_balancer_try_6_with_gb_model_trained_with_lung_more_fixed_predictions.txt')

        # AFTER FIXING PREDICTIONS
        # Experiment('breast_without_load_balancer', 'Breast (without load balancer)', plot_predictions=False, only_consider_intersection=True),
        # Experiment('breast_with_load_balancer_fixed_predictions', 'Breast fixed predictions (G.B. load balancer first model\n Trained model "svm_first_attempt")',
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
                   'Breast Sequential K-means log-likelihood',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('clustering/breast/logs_breast_clustering_n_stars_k_means_log_likelihood_2_clusters_1_core', 'Breast N-Stars K-means log-likelihood',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('clustering/breast/logs_breast_clustering_binpacking_k_means_log_likelihood_2_clusters_1_core', 'Breast Load balancer (v1) K-means log-likelihood',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment('clustering/breast/logs_breast_clustering_new_binpacking_model_k_means_log_likelihood_2_clusters_1_core',
                   'Breast Load balancer (full) K-means log-likelihood',
                   plot_predictions=True, only_consider_intersection=True),
        Experiment(
            'clustering/breast/logs_breast_clustering_overfitting_binpacking_model_k_means_log_likelihood_2_clusters_1_core',
            'Breast Load balancer (overfitting) K-means log-likelihood',
            plot_predictions=True, only_consider_intersection=True),

        # Breast, SVM, Linear
        Experiment('svm/breast/logs_breast_svm_sequential_kernel_linear_regression_1_core_6g',
                   'Breast Sequential SVM Linear',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/breast/logs_breast_svm_n_stars_kernel_linear_regression_1_core', 'Breast N-Stars SVM Linear',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_binpacking_kernel_linear_regression_1_core', 'Breast Load balancer (v1) SVM Linear (N_JOBS = -1)',
                   plot_predictions=True, only_consider_intersection=True),
        # TODO: add new_binpacking_model when finished in LIDI cluster
        # TODO: add overfitting_binpacking_model when finished in LIDI cluster

        # Breast, SVM, Poly
        Experiment('svm/breast/logs_breast_svm_sequential_kernel_poly_regression_1_core_6g',
                   'Breast Sequential SVM Poly',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/breast/logs_breast_svm_n_stars_kernel_poly_regression_1_core', 'Breast N-Stars SVM Poly',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_binpacking_kernel_poly_regression_1_core',
                   'Breast Load balancer (v1) SVM Poly (N_JOBS = -1)',
                   plot_predictions=True, only_consider_intersection=True),
        # TODO: add new_binpacking_model when finished in LIDI cluster
        # TODO: add overfitting_binpacking_model when finished in LIDI cluster

        # Breast, SVM, RBF
        Experiment('svm/breast/logs_breast_svm_sequential_kernel_rbf_regression_1_core_6g',
                   'Breast Sequential SVM RBF',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/breast/logs_breast_svm_n_stars_kernel_rbf_regression_1_core', 'Breast N-Stars SVM RBF',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/breast/logs_breast_svm_binpacking_kernel_rbf_regression_1_core',
                   'Breast Load balancer (v1) SVM RBF (N_JOBS = -1)',
                   plot_predictions=True, only_consider_intersection=True),
        # TODO: add new_binpacking_model when finished in LIDI cluster
        # TODO: add overfitting_binpacking_model when finished in LIDI cluster

        # Kidney, Clustering, K-means, 2 clusters
        Experiment('clustering/kidney/logs_kidney_clustering_sequential_k_means_log_likelihood_2_clusters_1_core_6g',
                   'Kidney Sequential K-means log-likelihood',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('clustering/kidney/logs_kidney_clustering_n_stars_k_means_log_likelihood_2_clusters_1_core',
                   'Kidney N-Stars K-means log-likelihood',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('clustering/kidney/logs_kidney_clustering_binpacking_k_means_log_likelihood_2_clusters_1_core',
                   'Kidney Load balancer (v1) K-means log-likelihood',
                   plot_predictions=True, only_consider_intersection=True),
        # TODO: add new_binpacking_model when finished in LIDI cluster
        # TODO: add overfitting_binpacking_model when finished in LIDI cluster

        # Kidney, SVM, Linear
        Experiment('svm/kidney/logs_kidney_svm_sequential_kernel_linear_regression_1_core_6g',
                   'Kidney Sequential SVM Linear',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/kidney/logs_kidney_svm_n_stars_kernel_linear_regression_1_core', 'Kidney N-Stars SVM Linear',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_binpacking_kernel_linear_regression_1_core',
                   'Kidney Load balancer (v1) SVM Linear (N_JOBS = -1)',
                   plot_predictions=True, only_consider_intersection=True),
        # TODO: add new_binpacking_model when finished in LIDI cluster
        # TODO: add overfitting_binpacking_model when finished in LIDI cluster

        # Kidney, SVM, Poly
        Experiment('svm/kidney/logs_kidney_svm_sequential_kernel_poly_regression_1_core_6g',
                   'Kidney Sequential SVM Poly',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/kidney/logs_kidney_svm_n_stars_kernel_poly_regression_1_core', 'Kidney N-Stars SVM Poly',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_binpacking_kernel_poly_regression_1_core',
                   'Kidney Load balancer (v1) SVM Poly (N_JOBS = -1)',
                   plot_predictions=True, only_consider_intersection=True),
        # TODO: add new_binpacking_model when finished in LIDI cluster
        # TODO: add overfitting_binpacking_model when finished in LIDI cluster

        # Kidney, SVM, RBF
        Experiment('svm/kidney/logs_kidney_svm_sequential_kernel_rbf_regression_1_core_6g',
                   'Kidney Sequential SVM RBF',
                   plot_predictions=False, only_consider_intersection=False, is_sequential=True),
        Experiment('svm/kidney/logs_kidney_svm_n_stars_kernel_rbf_regression_1_core', 'Kidney N-Stars SVM RBF',
                   plot_predictions=False, only_consider_intersection=True),
        Experiment('svm/kidney/logs_kidney_svm_binpacking_kernel_rbf_regression_1_core',
                   'Kidney Load balancer (v1) SVM RBF (N_JOBS = -1)',
                   plot_predictions=True, only_consider_intersection=True),
        # TODO: add new_binpacking_model when finished in LIDI cluster
        # TODO: add overfitting_binpacking_model when finished in LIDI cluster
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
    if not PLOT_IMAGES and not SAVE_IMGS:
        raise Exception('PLOT_IMAGES and SAVE_IMGS cannot be both False.')
    main()
