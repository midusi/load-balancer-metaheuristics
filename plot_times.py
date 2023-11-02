import logging
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import List, Optional, cast
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

TIMES_FOLDER_PATH = './Times results'


def save_dataset_img(dataset: str, sufix: str):
    dataset_fig_path = dataset.rstrip('.json') + sufix + '.png'
    final_path = os.path.join(TIMES_FOLDER_PATH, dataset_fig_path)
    plt.savefig(final_path)
    print(f'Saved "{final_path}"')


def store_mean_and_std(values_y: List[float], y: List[float], y_err: List[float], y_std_err: List[float]):
    """
    Computes the mean and std of Y values and stores them in y and y_err respectively
    :param values_y: Values to compute mean and std
    :param y: Y list where means are stored
    :param y_err: List where std are stored
    :param y_std_err: List where std / sqrt(n) are stored
    """
    std_error = np.std(values_y, ddof=1) / np.sqrt(len(values_y))
    mean = cast(float, np.mean(values_y))
    std_error = cast(float, std_error)
    std_error = round(std_error, 3)
    mean = round(mean, 3)
    std = cast(float, np.std(values_y))
    std = round(std, 3)

    y.append(mean)
    y_err.append(std)
    y_std_err.append(std_error)


def plot_dataset(title: str, dataset: str, time_field: str = 'execution_times', fix_min_max: bool = True, save_fig: bool = False):
    """
    Plots time data
    :param title: Chart title
    :param dataset: JSON dataset to get the data
    :param time_field: Time field to show in chart. Default = 'execution_times'
    :param fix_min_max: If True fixes the min and max Y value to better showing. Useful for 'execution_times' \
    as time field
    """
    # Reads and parses JSON results
    dataset_path = os.path.join(TIMES_FOLDER_PATH, dataset)
    with open(dataset_path, 'r') as file:
        content = json.loads(file.read())

    n_features = np.array(content['n_features'])
    execution_times = np.array(content[time_field])
    # idle_times = content['idle_times']  # TODO: check if needed

    # Sorts by number of features ascending preserving the order in all the lists
    arr1inds = n_features.argsort()
    n_features = n_features[arr1inds[::]]
    execution_times = execution_times[arr1inds[::]]

    x: List[int] = []
    last_n: Optional[int] = None
    y: List[float] = []
    values_y = []
    y_err: List[float] = []
    y_std_err: List[float] = []

    scatter_x = []
    scatter_y = []

    # Computes X, Y and error values
    for (cur_n, cur_y) in zip(n_features, execution_times):
        if np.isnan(cur_n) or np.isnan(cur_y):
            logging.warning(f'Found a NaN! X = {cur_n} | Y = {cur_y}')

        scatter_x.append(cur_n)
        scatter_y.append(cur_y)

        if cur_n != last_n:
            last_n = cur_n
            x.append(cur_n)

            # If it's not the first case, computes the mean and std
            if len(values_y) > 0:
                store_mean_and_std(values_y, y, y_err, y_std_err)
                values_y = []
        else:
            values_y.append(cur_y)

    # Stores last iteration
    store_mean_and_std(values_y, y, y_err, y_std_err)

    # Replaces NaNs with mean
    y_avg = np.nanmean(y)
    inds = np.where(np.isnan(y))
    y = np.array(y)
    y[inds] = y_avg
    y = y.tolist()

    # Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(scatter_x, scatter_y)
    plt.title(f'{title} | scatter')
    plt.xlabel("Number of features")
    plt.ylabel("Time (seconds)")

    if save_fig:
        save_dataset_img(dataset, f'_scatter ({time_field})')

    # Sets the labels min/max values
    min_y = min(np.min(y), 12)
    max_y = max(np.max(y), 32)

    # Fits a linear model
    reshaped = np.array(x).reshape((-1, 1))
    model = LinearRegression().fit(reshaped, y)
    y_pred_linear = model.predict(reshaped)

    # Fits a Polynomial model
    x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(reshaped)
    poly_model = LinearRegression().fit(x_, y)
    y_pred_poly = poly_model.predict(x_)

    # Plot with std
    # fig, ax = plt.subplots()
    # ax.errorbar(x, y, yerr=y_err, capsize=4)
    # plt.plot(x, y_pred_linear, label='Linear')
    # plt.plot(x, y_pred_poly, label='Polynomial')
    # plt.legend()
    # plt.ylim(min_y, max_y)
    # plt.title(f'{title} with std')
    # plt.xlabel("Number of features")
    # plt.ylabel("Time (seconds)")

    # Plot with std_error (https://www.statology.org/error-bars-python/)
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=y_std_err, capsize=4)
    plt.plot(x, y_pred_linear, label='Linear')
    plt.plot(x, y_pred_poly, label='Polynomial')
    plt.legend()
    if fix_min_max:
        plt.ylim(min_y, max_y)
    plt.title(f'{title} with std "normalized" (using field "{time_field}")')
    plt.xlabel("Number of features")
    plt.ylabel("Time (seconds)")

    if save_fig:
        save_dataset_img(dataset, f'_std_normalized ({time_field})')


def plot_time_by_iteration(title: str, dataset: str, save_fig: bool = False):
    # Reads and parses JSON results
    dataset_path = os.path.join(TIMES_FOLDER_PATH, dataset)
    with open(dataset_path, 'r') as file:
        content = json.loads(file.read())

    n_features = np.array(content['n_features'])
    time_by_iteration = np.array(content['times_by_iteration'])

    # Sorts by number of features ascending preserving the order in all the lists
    arr1inds = n_features.argsort()
    n_features = n_features[arr1inds[::]]
    time_by_iteration = time_by_iteration[arr1inds[::]]

    scatter_x = []
    scatter_y = []

    # Computes X, Y and error values
    for (cur_n, cur_y) in zip(n_features, time_by_iteration):
        scatter_x.append(cur_n)
        scatter_y.append(cur_y)

    # Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(scatter_x, scatter_y)
    plt.title(f'{title} (times by iteration)')
    plt.xlabel("Number of features")
    plt.ylabel("Time (seconds)")

    if save_fig:
        save_dataset_img(dataset, '_time_by_iteration')


def plot_num_of_iteration(title: str, dataset: str, save_fig: bool = False):
    """Plots the number of iterations (Y) by number of features (X)"""
    # Reads and parses JSON results
    dataset_path = os.path.join(TIMES_FOLDER_PATH, dataset)
    with open(dataset_path, 'r') as file:
        content = json.loads(file.read())

    n_features = np.array(content['n_features'])
    num_of_iterations = np.array(content['num_of_iterations'])

    # Sorts by number of features ascending preserving the order in all the lists
    arr1inds = n_features.argsort()
    n_features = n_features[arr1inds[::]]
    num_of_iterations = num_of_iterations[arr1inds[::]]

    scatter_x = []
    scatter_y = []

    # Computes X, Y and error values
    for (cur_n, cur_y) in zip(n_features, num_of_iterations):
        scatter_x.append(cur_n)
        scatter_y.append(cur_y)

    # Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(scatter_x, scatter_y)
    plt.title(f'{title} (number of iterations)')
    plt.xlabel("Number of features")
    plt.ylabel("Number of iterations")

    if save_fig:
        save_dataset_img(dataset, '_number_of_by_iteration')


def main():
    # Explicaciones de cada JSON en el archivo README.txt dentro de la carpeta 'Times results'
    # plot_dataset('200 stars by iteration (30 iterations)', '200_stars_30_it_2022-04-01_05_18_48_times.json')
    # plot_dataset('Sequential (30 iterations)', 'Sequential_30_it_2022-03-30_20_55_03_times.json')
    # plot_dataset('Complete (30 iterations)', 'Completo_2022-03-24_15_09_59_times.json')
    # plot_dataset('Sequential with max_iter = 10 in SVM (30 iterations)', 'Seq_SVM_10_max_iter_30_it_2022-04-05_04_24_25_times.json')
    # plot_dataset('With external process (30 iterations)', 'With_process_30_it_2022-04-08_02_16_34_times.json')
    # plot_dataset('With external process (30 iterations). Logs with DateStamp', 'With_process_30_it_datestamps_2022-04-09_09_24_41_times.json')
    # plot_dataset('With external process (30 iterations). G1GC', 'With_process_G1GC_2022-04-10_03_02_22_times.json')
    # plot_dataset('Con broadcast', 'Broadcast_2022-04-13_05_59_44_times.json')
    # plot_dataset('Sin excepciones para evitar NaNs', 'Sin_excepciones_2022-04-13_15_34_47_times.json')
    # plot_dataset('Sin excepciones para evitar NaNs y con DF en disco', 'Con_DF_en_disco_5_it_2022-04-13_18_47_18_times.json')
    # plot_dataset('Sin excepciones, nro de features randomizado', 'Randomized_30_it_2022-04-14_11_07_34_times.json')
    # plot_dataset('Sin excepciones, nro de features randomizado, DF en disco', 'Randomized_30_it_df_on_disk_2022-04-15_11_31_52_times.json')
    # plot_dataset('Sleep 1 seg', 'Sleep_1_5_it_2022-04-19_14_13_43_times.json')
    # plot_dataset('N_JOBS = 1', 'N_JOBS_1_5_it_2022-04-19_20_25_18_times.json')
    # plot_dataset('N_JOBS = 1', 'N_JOBS_1_5_it_2022-04-19_20_25_18_times.json')
    # plot_dataset('N_JOBS = 1 ascending', 'N_JOBS_1_ascending_2022-04-20_19_53_08_times.json')
    # plot_dataset('N_JOBS = 1 ascending, max_iter=10', 'N_JOBS_1_ascending_max_iter_1_2022-04-21_18_51_06_times.json')
    # plot_dataset('2 workers, no random state (30 iterations)', '2_workers_no_random_state_30_it_2022-04-22_11_44_44_times.json')
    # plot_dataset('With LockFile (5 iterations)', 'With_lock_5_it_2022-05-06_20_05_26_times.json')
    # plot_dataset('With LockFile absolute path (5 iterations)', 'With_locker_5_it_absolute_path_2022-05-09_21_00_14_times.json')
    # plot_dataset('With LockFile absolute path (30 iterations)', 'With_lock_30_it_2022-05-10_09_08_28_times.json')
    # plot_dataset('With LockFile, features randomized (30 iterations)', 'With_lock_randomized_30_it_2022-05-14_03_35_55_times.json')
    # plot_dataset('With times by iterations (30 iterations)', 'Times_by_iteration_30_it_2022-06-28_09_35_57_times.json')
    # plot_time_by_iteration('With times by iterations (30 iterations)', 'Times_by_iteration_30_it_2022-06-28_09_35_57_times.json')
    # plot_time_by_iteration('With times by iterations and test (3 iterations)', 'Times_by_iteration_and_test_3_it_2022-06-28_16_04_29_times.json')
    # plot_dataset(
    #     'Test time (3 iterations)',
    #     'Times_by_iteration_and_test_3_it_2022-06-28_16_04_29_times.json',
    #     time_field='test_times',
    #     fix_min_max=False
    # )
    # plot_time_by_iteration('Test time (30 iterations)',
    #                        'Times_by_iteration_and_test_30_it_2022-06-29_05_52_02_times.json')
    # plot_dataset(
    #     'Test time (30 iterations)',
    #     'Times_by_iteration_and_test_30_it_2022-06-29_05_52_02_times.json',
    #     time_field='test_times',
    #     fix_min_max=False
    # )

    # Times_by_iteration_and_num_of_iter_3_it_2022-06-30_14_54_45_times.json
    # plot_time_by_iteration('With num of it (3 iterations)',
    #                        'Times_by_iteration_and_num_of_iter_3_it_2022-06-30_14_54_45_times.json')

    # plot_num_of_iteration('With num of it (3 iterations)',
    #                        'Times_by_iteration_and_num_of_iter_3_it_2022-06-30_14_54_45_times.json')
    # plot_dataset(
    #     'With num of it (3 iterations)',
    #     'Times_by_iteration_and_num_of_iter_3_it_2022-06-30_14_54_45_times.json',
    #     time_field='test_times',
    #     fix_min_max=False
    # )

    # Times and metrics

    # Optimizer = AVLTree | Kernel = cosine
    # title = 'Optimizer = AVLTree | Kernel = cosine (30 iter.)'
    # json_path = 'Times_30_it_diff_svms_training_fitness/optimizer_avltree_kernel_cosine_30_it_with_training.json'
    # title_c_index = 'C-index | Optimizer = AVLTree | Kernel = cosine (30 iter.)'
    # json_path_c_index = 'Times_30_it_diff_svms_training_fitness/Logs/logs_times_optimizer_avltree_kernel_cosine_30_it_with_training_fitness.json'

    # Optimizer = AVLTree | Kernel = linear
    # title = 'Optimizer = AVLTree | Kernel = linear (30 iter.)'
    # json_path = 'Times_30_it_diff_svms_training_fitness/optimizer_avltree_kernel_linear_30_it_with_training.json'
    # title_c_index = 'C-index | Optimizer = AVLTree | Kernel = linear (30 iter.)'
    # json_path_c_index = 'Times_30_it_diff_svms_training_fitness/Logs/logs_times_optimizer_avltree_kernel_linear_30_it_with_training_fitness.json'

    # Optimizer = AVLTree | Kernel = poly
    # title = 'Optimizer = AVLTree | Kernel = poly (30 iter.)'
    # json_path = 'Times_30_it_diff_svms_training_fitness/optimizer_avltree_kernel_poly_30_it_with_training.json'
    # title_c_index = 'C-index | Optimizer = AVLTree | Kernel = poly (30 iter.)'
    # json_path_c_index = 'Times_30_it_diff_svms_training_fitness/Logs/logs_times_optimizer_avltree_kernel_poly_30_it_with_training_fitness.json'

    # Optimizer = AVLTree | Kernel = rbf
    title = 'Optimizer = AVLTree | Kernel = rbf (30 iter.)'
    json_path = 'Times_30_it_diff_svms_training_fitness/optimizer_avltree_kernel_rbf_30_it_with_training.json'
    title_c_index = 'C-index | Optimizer = AVLTree | Kernel = rbf (30 iter.)'
    json_path_c_index = 'Times_30_it_diff_svms_training_fitness/Logs/logs_times_optimizer_avltree_kernel_rbf_30_it_with_training_fitness.json'

    # Optimizer = AVLTree | Kernel = sigmoid
    # title = 'Optimizer = AVLTree | Kernel = sigmoid (30 iter.)'
    # json_path = 'Times_30_it_diff_svms_training_fitness/optimizer_avltree_kernel_sigmoid_30_it_2022-07-19_06_07_52_times.json'
    # TODO: it doesn't work. Too much NaNs values
    # title_c_index = 'C-index | Optimizer = AVLTree | Kernel = sigmoid (30 iter.)'
    # json_path_c_index = 'Times_30_it_diff_svms_training_fitness/Logs/logs_times_optimizer_avltree_kernel_sigmoid_30_it_fitness.json'

    # Optimizer = RBTree | Kernel = cosine
    # title = 'Optimizer = RBTree | Kernel = cosine (30 iter.)'
    # json_path = 'Times_30_it_diff_svms_training_fitness/optimizer_rbtree_kernel_cosine_30_it_2022-08-01_18_02_12_times.json'
    # title_c_index = 'C-index | Optimizer = RBTree | Kernel = cosine (30 iter.)'
    # json_path_c_index = 'Times_30_it_diff_svms_training_fitness/Logs/logs_times_optimizer_rbtree_kernel_cosine_30_it_fitness.json'

    # Optimizer = RBTree | Kernel = linear
    # title = 'Optimizer = RBTree | Kernel = linear (30 iter.)'
    # json_path = 'Times_30_it_diff_svms_training_fitness/optimizer_rbtree_kernel_linear_30_it_2022-07-22_23_39_13_times.json'
    # title_c_index = 'C-index | Optimizer = RBTree | Kernel = linear (30 iter.)'
    # json_path_c_index = 'Times_30_it_diff_svms_training_fitness/Logs/logs_times_optimizer_rbtree_kernel_linear_30_it_fitness.json'

    # Optimizer = RBTree | Kernel = poly
    # title = 'Optimizer = RBTree | Kernel = poly (30 iter.)'
    # json_path = 'Times_30_it_diff_svms_training_fitness/optimizer_rbtree_kernel_poly_30_it_2022-07-23_18_56_03_times.json'
    # title_c_index = 'C-index | Optimizer = RBTree | Kernel = poly (30 iter.)'
    # json_path_c_index = 'Times_30_it_diff_svms_training_fitness/Logs/logs_times_optimizer_rbtree_kernel_poly_30_it_fitness.json'

    # Optimizer = RBTree | Kernel = rbf
    # title = 'Optimizer = RBTree | Kernel = rbf (30 iter.)'
    # json_path = 'Times_30_it_diff_svms_training_fitness/optimizer_rbtree_kernel_rbf_30_it_2022-07-25_13_25_06_times.json'
    # title_c_index = 'C-index | Optimizer = RBTree | Kernel = rbf (30 iter.)'
    # json_path_c_index = 'Times_30_it_diff_svms_training_fitness/Logs/logs_times_optimizer_rbtree_kernel_rbf_30_it_fitness.json'

    # Optimizer = RBTree | Kernel = sigmoid
    # title = 'Optimizer = RBTree | Kernel = sigmoid (30 iter.)'
    # json_path = 'Times_30_it_diff_svms_training_fitness/optimizer_rbtree_kernel_sigmoid_30_it_2022-07-29_08_39_13_times.json'
    # TODO: it doesn't work. Too much NaNs values
    # title_c_index = 'C-index | Optimizer = RBTree | Kernel = sigmoid (30 iter.)'
    # json_path_c_index = 'Times_30_it_diff_svms_training_fitness/Logs/logs_times_optimizer_rbtree_kernel_sigmoid_30_it_fitness.json'

    # Times
    save_img = False
    # plot_dataset(title, json_path, save_fig=save_img)
    # plot_dataset(f'Test time {title}', json_path, time_field='test_times', fix_min_max=False, save_fig=save_img)
    plot_dataset(f'Training fitness {title}', json_path, time_field='train_scores', fix_min_max=False, save_fig=save_img)
    # plot_time_by_iteration(f'With num of it {title}', json_path, save_fig=save_img)
    # C-Index
    # plot_dataset(title_c_index, json_path_c_index, time_field='fitness', fix_min_max=False, save_fig=save_img)

    plt.show()


if __name__ == '__main__':
    main()
