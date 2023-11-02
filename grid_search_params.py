import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from load_balancer_model import get_x_and_y_clustering, get_x_and_y_svm, get_x_and_y_rf

# TRAIN: bool = True
TRAIN: bool = False

TRAINING_CSV_PATH_CLUSTERING = 'LoadBalancerDatasets/ClusteringTimesRecord-2023-08-18.csv'

def __fit_predict_and_plot(x_train: np.ndarray, y_train: np.ndarray, x_test_axis: np.ndarray,
                           test_data: np.ndarray, model):
    """Fits model with x and y, prints predictions with test_data and plots a scatter plot."""
    model.fit(x_train, y_train)

    # Creates a new figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plots the predictions for train data
    number_of_features_train = x_train[:, 0]
    predictions_train = model.predict(x_train)
    print('predictions_train:')
    print(predictions_train)
    ax1.title.set_text('Train data')
    ax1.scatter(number_of_features_train, y_train, label='Real values')
    ax1.scatter(number_of_features_train, predictions_train, label='Predictions', color='red')
    ax1.set_xlabel('Number of features')
    ax1.set_ylabel('Execution time (s)')
    ax1.legend()

    # Plots the predictions with test_data
    predictions_test = model.predict(test_data)
    print('predictions_test:')
    print(predictions_test)
    ax2.title.set_text('Test data')
    ax2.scatter(x_test_axis, predictions_test, label='Predictions')
    ax2.set_xlabel('Number of features')
    ax2.set_ylabel('Execution time (s)')
    ax2.legend()


    plt.show()


def grid_search_gradient_booster(x: np.ndarray, y: np.ndarray):
    """Makes a GridSearch for the GradientBoostingRegressor algorithm."""
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'min_samples_split': [2, 3, 4, 5, 6],
        'min_samples_leaf': [10, 20, 30, 40, 50]
    }
    model = GradientBoostingRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(x, y)
    print('Best params:')
    print(grid_search.best_params_)


def grid_search_hist_gradient_booster(x: np.ndarray, y: np.ndarray):
    """Makes a GridSearch for the HistGradientBoostingRegressor algorithm."""
    param_grid = {
        'max_iter': [100, 200, 300, 400, 500],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_leaf_nodes': [31, 41, 51, 61, 71],
        'min_samples_leaf': [10, 20, 30, 40, 50]
    }
    model = HistGradientBoostingRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(x, y)
    print(grid_search.best_params_)


def grid_search_svm_poly(x: np.ndarray, y: np.ndarray):
    """Makes a GridSearch for the SVR algorithm with poly kernel."""
    param_grid = {
        'kernel': ['poly'],
        'degree': [2, 3],
        'C': [0.1, 0.5, 1]
    }
    model = SVR()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(x, y)
    print(grid_search.best_params_)


def grid_search_svm_rbf_sigmoid(x: np.ndarray, y: np.ndarray):
    """Makes a GridSearch for the SVR algorithm."""
    param_grid = {
        'kernel': ['rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'C': [0.1, 0.5, 1, 2, 3, 4, 5]
    }
    model = SVR()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(x, y)
    print(grid_search.best_params_)


def train_with_grid_search(x: np.ndarray, y: np.ndarray):
    print('HistGradientBoostingRegressor:')
    start = time.time()
    grid_search_hist_gradient_booster(x, y)
    print(f'Time grid_search_hist_gradient_booster: {time.time() - start}')

    # print('GradientBoostingRegressor:')
    # start = time.time()
    # grid_search_gradient_booster(x, y)
    # print(f'Time grid_search_gradient_booster: {time.time() - start}')

    # print('SVM Poly:')
    # start = time.time()
    # grid_search_svm_poly(x, y)
    # print(f'Time svm: {time.time() - start}')

    # print('SVM RBF and sigmoid:')
    # start = time.time()
    # grid_search_svm_rbf_sigmoid(x, y)
    # print(f'Time svm: {time.time() - start}')


def test_best_models_svm(x: np.ndarray, y: np.ndarray, number_of_features: np.ndarray, data: np.ndarray):
    """Tests best models computed previously"""
    # HistGradientBoostingRegressor
    best_hist_params = {'learning_rate': 0.2, 'max_depth': 4, 'max_iter': 300, 'max_leaf_nodes': 41,
                        'min_samples_leaf': 20}
    model = HistGradientBoostingRegressor(**best_hist_params)
    __fit_predict_and_plot(x, y, number_of_features, data, model)


def test_best_models_rf(x: np.ndarray, y: np.ndarray, number_of_features: np.ndarray, data: np.ndarray):
    """Tests best models computed previously"""
    # HistGradientBoostingRegressor
    best_hist_params = {'learning_rate': 0.01, 'max_depth': 2, 'max_iter': 400, 'max_leaf_nodes': 31,
                        'min_samples_leaf': 40}
    model = HistGradientBoostingRegressor(**best_hist_params)
    __fit_predict_and_plot(x, y, number_of_features, data, model)


def test_best_models_clustering(x: np.ndarray, y: np.ndarray, number_of_features: np.ndarray, data: np.ndarray):
    """Tests best models computed previously"""
    # HistGradientBoostingRegressor
    best_hist_params = {'learning_rate': 0.01, 'max_depth': 6, 'max_iter': 400, 'max_leaf_nodes': 41,
                        'min_samples_leaf': 10}
    model = HistGradientBoostingRegressor(**best_hist_params)
    __fit_predict_and_plot(x, y, number_of_features, data, model)

    # GradientBoostingRegressor
    # best_grad_boost_params = {'learning_rate': 0.01, 'max_depth': 6, 'min_samples_leaf': 10, 'min_samples_split': 3,
    #                           'n_estimators': 200}
    # model = GradientBoostingRegressor(**best_grad_boost_params)
    # __fit_predict_and_plot(x, y, number_of_features, data, model)

    # SVM Poly
    # best_svm_poly_params = {'C': 1, 'degree': 3, 'kernel': 'poly'}
    # model = SVR(**best_svm_poly_params)
    # __fit_predict_and_plot(x, y, number_of_features, data, model)


def get_test_data_for_clustering():
    """
    Generates a random dataset using the structure [number_of_features, 1082, 2, 1, 2] and dtype = np.float32.
    Where number of features should be generated in order from 10 to 100 and then from 100 to 25000 samples.
    This struct is only for Clustering.
    """
    data_10_to_100 = np.array([i for i in range(10, 110, 10)])
    data_100_to_20000 = np.array([i for i in range(100, 20100, 100)])
    data = np.concatenate((data_10_to_100, data_100_to_20000))
    number_of_elements = data.shape[0]
    data = data.reshape((number_of_elements, 1))
    data = np.concatenate((data, np.full((number_of_elements, 1), 1082)), axis=1)
    data = np.concatenate((data, np.full((number_of_elements, 1), 2)), axis=1)
    data = np.concatenate((data, np.full((number_of_elements, 1), 1)), axis=1)
    data = np.concatenate((data, np.full((number_of_elements, 1), 2)), axis=1)
    data = data.astype(np.float32)
    print('Test data:')
    print(data)

    return data



def get_test_data_for_rf():
    """
    Generates a random dataset using the structure [number_of_features, 1082, 2, 1, 2] and dtype = np.float32.
    Where number of features should be generated in order from 10 to 100 and then from 100 to 25000 samples.
    This struct is only for RF.
    """
    data_10_to_100 = np.array([i for i in range(10, 110, 10)])
    data_100_to_20000 = np.array([i for i in range(100, 20100, 100)])
    data = np.concatenate((data_10_to_100, data_100_to_20000))
    number_of_elements = data.shape[0]
    data = data.reshape((number_of_elements, 1))
    data = np.concatenate((data, np.full((number_of_elements, 1), 1082)), axis=1)
    data = np.concatenate((data, np.full((number_of_elements, 1), 10)), axis=1)  # Number of Trees
    data = data.astype(np.float32)
    print('Test data:')
    print(data)

    return data



def get_test_data_for_svm():
    """
    Generates a random dataset using the structure [number_of_features, 1082, 2, 1, 2] and dtype = np.float32.
    Where number of features should be generated in order from 10 to 100 and then from 100 to 25000 samples.
    This struct is only for SVM.
    """
    data_10_to_100 = np.array([i for i in range(10, 110, 10)])
    data_100_to_20000 = np.array([i for i in range(100, 20100, 100)])
    data = np.concatenate((data_10_to_100, data_100_to_20000))
    number_of_elements = data.shape[0]
    data = data.reshape((number_of_elements, 1))
    data = np.concatenate((data, np.full((number_of_elements, 1), 1082)), axis=1)
    data = np.concatenate((data, np.full((number_of_elements, 1), 1)), axis=1)  # Optimizer = 'avltree'
    data = np.concatenate((data, np.full((number_of_elements, 1), 0)), axis=1)  # Kernel = 'linear'
    data = data.astype(np.float32)
    print('Test data:')
    print(data)

    return data


def main():
    # Clustering
    def clustering():
        data = get_test_data_for_clustering()
        number_of_features = data[:, 0].copy()  # Gets number of features from data (without MinMax transformation)
        x_clustering, _x_clustering_without_min_max, y_clustering, _ord_encoder_clustering, min_max_scaler_clustering = get_x_and_y_clustering()
        if TRAIN:
            train_with_grid_search(x_clustering, y_clustering)
        else:
            data[:, [0, 1]] = min_max_scaler_clustering.transform(data[:, [0, 1]])
            print('Test data after MinMaxScaling by clustering:')
            print(data)
            test_best_models_clustering(x_clustering, y_clustering, number_of_features, data)

    # SVM
    def svm():
        data = get_test_data_for_svm()
        number_of_features = data[:, 0].copy()  # Gets number of features from data (without MinMax transformation)
        x_svm, _x_svm_without_min_max, y_svm, _ord_encoder_svm, min_max_scaler_svm = get_x_and_y_svm()
        if TRAIN:
            train_with_grid_search(x_svm, y_svm)
        else:
            data[:, [0, 1]] = min_max_scaler_svm.transform(data[:, [0, 1]])
            print('Test data after MinMaxScaling by SVM:')
            print(data)
            test_best_models_svm(x_svm, y_svm, number_of_features, data)

    def rf():
        data = get_test_data_for_rf()
        number_of_features = data[:, 0].copy()  # Gets number of features from data (without MinMax transformation)
        x_rf, _x_svm_without_min_max, y_rf, _ord_encoder_rf, min_max_scaler_rf = get_x_and_y_rf()
        if TRAIN:
            train_with_grid_search(x_rf, y_rf)
        else:
            data[:, [0, 1]] = min_max_scaler_rf.transform(data[:, [0, 1]])
            print('Test data after MinMaxScaling by RF:')
            print(data)
            test_best_models_rf(x_rf, y_rf, number_of_features, data)

    clustering()
    svm()
    rf()
if __name__ == '__main__':
    main()
