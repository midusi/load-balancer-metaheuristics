# TODO: probar actualizando la libreria a ver si el fitness durante training cambia, y ver si siguen saliendo muchos NaNs para el kernel Sigmoid
import json
import os
import time
from typing import Optional, cast, Dict, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sksurv.svm import FastKernelSurvivalSVM
import matplotlib.pyplot as plt
from core import fitness_function_with_checking_sequential
from utils import read_survival_data

# Folder where results are stored
TIMES_FOLDER_PATH = 'CV_Experiments'

# To replicate randomness
RANDOM_STATE: Optional[int] = None

# Number of iterations to run and store
NUMBER_OF_ITERATIONS: int = 2

# If specified, this sets a threshold of maximum number of features to test. If None, all features is the maximum
MAX_NUMBER_OF_FEATURES: Optional[int] = None

Keys = Literal['fit_times', 'train_scores', 'test_scores']


def plot(title: str, dataset_path: str, field: Keys, y_label: Optional[str] = None):
    # Reads and parses JSON results
    with open(dataset_path, 'r') as file:
        content = json.loads(file.read())

    n_features = np.array(content['n_features'])
    time_by_iteration = np.array(content[field])

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
    plt.title(title)
    plt.xlabel("Number of features")
    plt.ylabel("Time (seconds)" if y_label is None else y_label)


def plot_all(json_dest: str):
    """Plots train scores, test scores and training time"""
    plot('Train score', json_dest, 'train_scores', y_label='Train fitness')
    plot('Test score', json_dest, 'test_scores', y_label='Test fitness')
    plot('Fit time', json_dest, 'fit_times')
    plt.show()


def compute_cross_validation(subset: pd.DataFrame, y: np.ndarray) -> Dict:
    """
    Computa una validacion cruzada calculando el accuracy.
    :param subset: Subset de features a utilizar en el RandomForest evaluado en el CrossValidation.
    :param y: Clases.
    :return: Promedio del accuracy obtenido en cada fold del CrossValidation.
    """
    # TODO: devolver numero de iteraciones y tiempo por iteraciones
    res = cross_validate(
        # FastKernelSurvivalSVM(rank_ratio=0.0, max_iter=1000, tol=1e-5, random_state=RANDOM_STATE),
        # FastKernelSurvivalSVM(rank_ratio=0.0, max_iter=1000, tol=1e-5),
        FastKernelSurvivalSVM(rank_ratio=1.0, max_iter=1000, tol=1e-5),
        subset,
        y,
        # cv=10,
        cv=3,
        return_train_score=True,
        return_estimator=False
        # n_jobs=N_JOBS
    )

    return res


def main():
    """
    Runs some experiments to check CrossValidation and Survival-SVM concepts
    """
    # Obtiene los datos necesarios de supervivencia
    x, y = read_survival_data(add_epsilon=True)  # TODO: check if needed. SVM should have this in True

    number_samples, number_features = x.shape
    step = 1000

    print(f'Survival dataset')
    print(f'\tSamples (rows) -> {number_samples} | Features (columns) -> {number_features}')
    print(f'\tY shape -> {y.shape}')

    total_n_features = MAX_NUMBER_OF_FEATURES if MAX_NUMBER_OF_FEATURES is not None else x.shape[1]

    # Final data
    fit_times = []
    train_scores = []
    test_scores = []
    number_of_features = []

    # Runs the iterations
    for i in range(NUMBER_OF_ITERATIONS):
        # current_n_features = step
        current_n_features = 10

        print(f'Iteration {i + 1}')
        while current_n_features <= total_n_features:
            random_features_to_select = np.zeros(total_n_features, dtype=int)
            random_features_to_select[:current_n_features] = 1
            np.random.shuffle(random_features_to_select)

            start_worker_time = time.time()
            cv_result = fitness_function_with_checking_sequential(
                compute_cross_validation,
                random_features_to_select,
                x,
                y
            )
            cv_result = cast(Dict, cv_result)

            # Time
            cur_exec_time = time.time() - start_worker_time
            cur_exec_time = round(cur_exec_time, 4)

            # Fitness
            fit_time = cv_result['fit_time'].mean()
            score_time = cv_result['score_time'].mean()
            train_score = cv_result['train_score'].mean()
            test_score = cv_result['test_score'].mean()

            # Adds to lists
            fit_times.append(round(fit_time, 4))
            train_scores.append(round(train_score, 4))
            test_scores.append(round(test_score, 4))
            number_of_features.append(current_n_features)

            print(f'{current_n_features} features:')
            print(f'\tCV took {cur_exec_time} seconds')
            print(f'\tFitness training: {train_score} | Fitness testing: {test_score}')  # TODO: complete
            print(f'\tTraining time: {fit_time} | Testing time: {score_time}')  # TODO: complete

            # current_n_features += step
            # current_n_features += step if current_n_features >= step else 10  # TODO: leave this line, remove below
            current_n_features += step if current_n_features >= 100 else 50

    # Saves times in JSON for post-processing
    now = time.strftime('%Y-%m-%d_%H_%M_%S')
    json_file = f'{now}_times.json'
    json_dest = os.path.join(TIMES_FOLDER_PATH, json_file)

    print(f'Saving lists in JSON format in {json_dest}')
    result_dict = {
        'n_features': number_of_features,
        'fit_times': fit_times,
        'train_scores': train_scores,
        'test_scores': test_scores,
    }

    with open(json_dest, 'w+') as file:
        file.write(json.dumps(result_dict))

    print('Saved.')

    plot_all(json_dest)


if __name__ == '__main__':
    # json_dest = os.path.join(TIMES_FOLDER_PATH, '2022-09-19_11_10_04_times.json')
    # plot_all(json_dest)
    main()
