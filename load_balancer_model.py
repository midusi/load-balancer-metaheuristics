import os.path
import joblib
from typing import Tuple, List, Optional, Literal
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import ensemble
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, MinMaxScaler
from plot_experiments import get_mean_and_std
from utils import ModelName

# Path where the trained models will be saved
TRAINED_MODELS_PATH_CLUSTERING = 'Trained_models/clustering'
TRAINED_MODELS_PATH_SVM = 'Trained_models/svm'
TRAINED_MODELS_PATH_RF = 'Trained_models/rf'

# Folder where the CSVs with the AWS training data are
TRAINING_CSV_PATH_CLUSTERING = 'LoadBalancerDatasets/ClusteringTimesRecord-2023-08-18.csv'
TRAINING_CSV_PATH_SVM = 'LoadBalancerDatasets/SVMTimesRecord-2023-08-18.csv'
TRAINING_CSV_PATH_RF = 'LoadBalancerDatasets/RFTimesRecord-2023-08-18.csv'

# Folder where the CSVs with the AWS + experiments data are
TRAINING_CSV_PATH_CLUSTERING_FULL = 'LoadBalancerDatasets/ClusteringTimesRecord_full-2023-10-24.csv'
TRAINING_CSV_PATH_SVM_FULL = 'LoadBalancerDatasets/SVMTimesRecord_full-2023-10-24.csv'

# Folder where the CSVs with the training + AWS data are
TRAINING_CSV_PATH_CLUSTERING_OVERFITTING = 'LoadBalancerDatasets/ClusteringTimesRecord_overfitting-2023-10-24.csv'
TRAINING_CSV_PATH_SVM_OVERFITTING = 'LoadBalancerDatasets/SVMTimesRecord_overfitting-2023-10-24.csv'

# Uses the full CSV with all the data. Otherwise, uses the CSV with the initial data (from Multiomix)
Strategy = Literal['aws', 'full', 'overfitting']
# DATASET_TO_USE: Strategy = 'full'
DATASET_TO_USE: Strategy = 'overfitting'
# DATASET_TO_USE: Strategy = 'aws'

# If True, plots X, Y, and all the trained models prediction
# PLOT_MODELS = False
PLOT_MODELS = False

# If True trains the model from scratch, otherwise uses the TRAINED_MODELS_PATH_X to retrieve all the models
# TRAIN_MODELS = True
TRAIN_MODELS = True

# If True, save the models .pkl. Only used if TRAIN_MODELS is True
SAVE_MODELS = True
# SAVE_MODELS = False


def get_x_and_y_clustering(dataset_to_use: Strategy) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[OrdinalEncoder], MinMaxScaler]:
    """
    Gets X and Y data from the training CSV to train the LoadBalancer for clustering models.
    :param dataset_to_use: Strategy to check which dataset to use. If 'full', uses the full CSV with all the data
    (training + AWS data). If 'overfitting', uses the CSV with only the training data. Otherwise, uses the CSV with the
    initial data (from Multiomix/AWS).
    :return: X, X (without MinMax scaler), Y, and OrdinalEncoder and MinMaxScaler instances
    """
    print('Getting data for Clustering')
    if dataset_to_use == 'full':
        file_path = TRAINING_CSV_PATH_CLUSTERING_FULL
    elif dataset_to_use == 'overfitting':
        file_path = TRAINING_CSV_PATH_CLUSTERING_OVERFITTING
    else:
        file_path = TRAINING_CSV_PATH_CLUSTERING

    print(f'Using "{file_path}" dataset (Strategy: "{dataset_to_use}")')

    df = pd.read_csv(file_path)

    # We don't need fitness result to train a time model! And task is always ranking for the moment. So, keeps
    # only the needed columns
    df = df[['Number of features', 'Number of samples', 'Algorithm', 'Number of clusters', 'Scoring method', 'Execution time']]

    # Sets dtypes for all the columns as number
    df = df.astype({'Number of features': 'int32', 'Number of samples': 'int32', 'Number of clusters': 'int32', 'Algorithm': 'category',
                    'Scoring method': 'category', 'Execution time': 'float64'})

    # Filters all the rows where 'Number of features' is 0
    df = df[df['Number of features'] != 0]

    # Encodes categorical features
    # NOTE: it's not needed as Multiomix already encodes them to numeric values
    ordinal_enc = None

    # Gets X without MinMax and Y
    class_column = 'Execution time'
    y = df.pop(class_column).values
    x_without_min_max = df.values

    # Scales number of features and samples
    numeric_features = ['Number of features', 'Number of samples']
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(df[numeric_features].values)  # Calling .values prevents warning https://stackoverflow.com/a/69378867/7058363
    df[numeric_features] = min_max_scaler.transform(df[numeric_features])

    # Prints min and max values of both columns
    print(f'Min and max values of {numeric_features[0]}: {min_max_scaler.data_min_[0]} and {min_max_scaler.data_max_[0]}')
    print(f'Min and max values of {numeric_features[1]}: {min_max_scaler.data_min_[1]} and {min_max_scaler.data_max_[1]}')

    x = df.values

    print(f'Used features: {", ".join(df.columns.values)}')
    print(f'Used Y: {class_column}')

    return x, x_without_min_max, y, ordinal_enc, min_max_scaler


def get_x_and_y_rf() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[OrdinalEncoder], MinMaxScaler]:
    """
    Gets X and Y data from the training CSV to train the LoadBalancer for Random Forest models.
    :return: X, X (without MinMax scaler), Y, and OrdinalEncoder and MinMaxScaler instances
    """
    print('Getting data for Random Forest')
    df = pd.read_csv(TRAINING_CSV_PATH_RF)

    # We don't need fitness result to train a time model! And task is always ranking for the moment. So, keeps
    # only the needed columns
    df = df[['Number of features', 'Number of samples', 'Number of trees', 'Execution time']]

    # Sets dtypes for all the columns as number
    df = df.astype({'Number of features': 'int32', 'Number of samples': 'int32', 'Number of trees': 'int32',
                    'Execution time': 'float64'})

    # Gets X without MinMax and Y
    class_column = 'Execution time'
    y = df.pop(class_column).values
    x_without_min_max = df.values

    # Scales number of features and samples
    numeric_features = ['Number of features', 'Number of samples']
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(df[numeric_features].values)  # Calling .values prevents warning https://stackoverflow.com/a/69378867/7058363
    df[numeric_features] = min_max_scaler.transform(df[numeric_features])
    x = df.values

    print(f'Used features: {", ".join(df.columns.values)}')
    print(f'Used Y: {class_column}')

    return x, x_without_min_max, y, None, min_max_scaler


def get_x_and_y_svm(dataset_to_use: Strategy) -> Tuple[np.ndarray, np.ndarray, np.ndarray, OrdinalEncoder, MinMaxScaler]:
    """
    Gets X and Y data from the training CSV to train the LoadBalancer for SVM models.
    :param dataset_to_use: Strategy to check which dataset to use. If 'full', uses the full CSV with all the data
    (training + AWS data). If 'overfitting', uses the CSV with only the training data. Otherwise, uses the CSV with the
    initial data (from Multiomix/AWS).
    :return: X, X (without MinMax scaler), Y, and OrdinalEncoder and MinMaxScaler instances
    """
    print('Getting data for SVM')
    if dataset_to_use == 'full':
        file_path = TRAINING_CSV_PATH_SVM_FULL
    elif dataset_to_use == 'overfitting':
        file_path = TRAINING_CSV_PATH_SVM_OVERFITTING
    else:
        file_path = TRAINING_CSV_PATH_SVM

    print(f'Using "{file_path}" dataset (Strategy: "{dataset_to_use}")')

    df = pd.read_csv(file_path)

    # We don't need fitness result to train a time model! And task is always ranking for the moment. So, keeps
    # only the needed columns
    df = df[['Number of features', 'Number of samples', 'Kernel', 'Optimizer', 'Execution time']]

    # Sets dtypes for all the columns as number except for 'Optimizer' column
    df = df.astype({'Number of features': 'int32', 'Number of samples': 'int32', 'Kernel': 'category',
                    'Optimizer': 'category', 'Execution time': 'float64'})

    # Encodes categorical features
    # NOTE: this is needed in the SVM model as we forgot to cast the 'Optimizer' column to category in the CSV
    # from Multiomix
    categorical_features = ['Optimizer']
    ordinal_enc = OrdinalEncoder()
    ordinal_enc.fit(df[categorical_features].values)  # Calling .values prevents warning https://stackoverflow.com/a/69378867/7058363
    df[categorical_features] = ordinal_enc.transform(df[categorical_features].values)

    # Gets X without MinMax and Y
    class_column = 'Execution time'
    y = df.pop(class_column).values
    x_without_min_max = df.values

    # Scales number of features and samples
    numeric_features = ['Number of features', 'Number of samples']
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(df[numeric_features].values)  # Calling .values prevents warning https://stackoverflow.com/a/69378867/7058363
    df[numeric_features] = min_max_scaler.transform(df[numeric_features])
    x = df.values

    print(f'Used features: {", ".join(df.columns.values)}')
    print(f'Used Y: {class_column}')

    return x, x_without_min_max, y, ordinal_enc, min_max_scaler


def test_model(model: RegressorMixin, x: np.ndarray, y: np.ndarray) -> RegressorMixin:
    """
    Executes a Cross-Validation with a specific model and returns the best model get during the process
    :param model: Model to fit and predict inside the CV
    :param x: X data
    :param y: Y Data
    :return: Best model obtained (i.e., lowest MSE obtained) and the same model trained without the MinMax scaler
    """
    result = cross_validate(model, x, y, cv=10, return_train_score=False, return_estimator=True,
                            scoring=['neg_mean_squared_error', 'r2'])

    best_model_idx = np.argmax(result['test_neg_mean_squared_error'])
    best_model = result['estimator'][best_model_idx]
    best_r2 = result['test_r2'][best_model_idx]
    best_mse = result['test_neg_mean_squared_error'][best_model_idx] * -1

    print(f'The model "{best_model}" has obtained a R2 = {best_r2} and a MSE = {best_mse}')
    return best_model


def train_models(model_name: ModelName):
    """Generates a lot of models using SVM training data"""
    if model_name == 'svm':
        x, x_without_min_max, y, ord_encoder, min_max_scaler = get_x_and_y_svm(DATASET_TO_USE)
        trained_models_path = TRAINED_MODELS_PATH_SVM
        # Best parameters obtained with GridSearchCV in 'grid_search_params.py'
        best_hist_params = {'learning_rate': 0.2, 'max_depth': 4, 'max_iter': 300, 'max_leaf_nodes': 41,
                            'min_samples_leaf': 20}
    elif model_name == 'clustering':
        x, x_without_min_max, y, ord_encoder, min_max_scaler = get_x_and_y_clustering(DATASET_TO_USE)
        trained_models_path = TRAINED_MODELS_PATH_CLUSTERING
        # Best parameters obtained with GridSearchCV in 'grid_search_params.py'
        best_hist_params = {'learning_rate': 0.01, 'max_depth': 6, 'max_iter': 400, 'max_leaf_nodes': 41,
                            'min_samples_leaf': 10}
    elif model_name == 'rf':
        x, x_without_min_max, y, ord_encoder, min_max_scaler = get_x_and_y_rf()
        trained_models_path = TRAINED_MODELS_PATH_RF
        # Best parameters obtained with GridSearchCV in 'grid_search_params.py'
        best_hist_params = {'learning_rate': 0.01, 'max_depth': 2, 'max_iter': 400, 'max_leaf_nodes': 31,
                            'min_samples_leaf': 40}
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    print(f'Training {model_name} with {len(y)} rows')

    # Saves encoders
    if SAVE_MODELS:
        # Adds a suffix in case training with full data (RF not implemented yet)
        if model_name != 'rf':
            if DATASET_TO_USE == 'full':
                trained_models_path += '_full'
            elif DATASET_TO_USE == 'overfitting':
                trained_models_path += '_overfitting'

        # Creates the target dir if it doesn't exist
        if os.path.isdir(trained_models_path):
            print(f'{trained_models_path} already exists. Change SAVE_MODELS to False or change the dest folder. '
                  f'Exiting...')
            exit(-1)

        print(f'{trained_models_path} does not exist. Creating...')
        mode = 0o777
        os.mkdir(trained_models_path, mode)
        os.chmod(trained_models_path, mode)  # Mode in mkdir is sometimes ignored: https://stackoverflow.com/a/5231994/7058363

        if ord_encoder:
            joblib.dump(ord_encoder, os.path.join(trained_models_path, 'ord_encoder.pkl'))
        joblib.dump(min_max_scaler, os.path.join(trained_models_path, 'min_max_scaler.pkl'))

    # LinearRegression
    print('LinearRegression')
    print('LinearRegression no MinMax')  # NOTE: this performs the same as with MinMax scaler, so keeps this variant as it's simpler
    linear_1_no_min_max_path = 'best_linear_model_no_min_max.pkl'
    if TRAIN_MODELS:
        linear_model = LinearRegression()
        best_linear_model_no_min_max = test_model(linear_model, x_without_min_max, y)
        if SAVE_MODELS:
            joblib.dump(best_linear_model_no_min_max, os.path.join(trained_models_path, linear_1_no_min_max_path))
    else:
        best_linear_model_no_min_max = joblib.load(os.path.join(trained_models_path, linear_1_no_min_max_path))

    print('LinearRegression (degree=2)')
    print('LinearRegression no MinMax (degree=2)')  # NOTE: this performs the same as with MinMax scaler, so keeps this variant as it's simpler
    linear_2_no_min_max_path = 'best_linear_model_2_no_min_max.pkl'
    x_polynomial_2_without_min_max = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x_without_min_max)
    if TRAIN_MODELS:
        linear_model_2 = LinearRegression()
        best_linear_model_2_no_min_max = test_model(linear_model_2, x_polynomial_2_without_min_max, y)
        if SAVE_MODELS:
            joblib.dump(best_linear_model_2_no_min_max, os.path.join(trained_models_path, linear_2_no_min_max_path))
    else:
        best_linear_model_2_no_min_max = joblib.load(os.path.join(trained_models_path, linear_2_no_min_max_path))

    print('LinearRegression (degree=3)')
    print('LinearRegression no MinMax (degree=3)')  # NOTE: this performs the same as with MinMax scaler, so keeps this variant as it's simpler
    linear_3_no_min_max_path = 'best_linear_model_3_no_min_max.pkl'
    x_polynomial_3_without_min_max = PolynomialFeatures(degree=3, include_bias=False).fit_transform(x_without_min_max)
    if TRAIN_MODELS:
        linear_model_3 = LinearRegression()
        best_linear_model_3_no_min_max = test_model(linear_model_3, x_polynomial_3_without_min_max, y)
        if SAVE_MODELS:
            joblib.dump(best_linear_model_3_no_min_max, os.path.join(trained_models_path, linear_3_no_min_max_path))
    else:
        best_linear_model_3_no_min_max = joblib.load(os.path.join(trained_models_path, linear_3_no_min_max_path))

    # HistGradientBoostingRegressor
    print('HistGradientBoostingRegressor')
    gradient_booster_path = 'best_gradient_booster_model.pkl'
    if TRAIN_MODELS:
        gradient_booster_model = ensemble.HistGradientBoostingRegressor(**best_hist_params)
        best_gradient_booster_model = test_model(gradient_booster_model, x, y)
        if SAVE_MODELS:
            joblib.dump(best_gradient_booster_model, os.path.join(trained_models_path, gradient_booster_path))
    else:
        best_gradient_booster_model = joblib.load(os.path.join(trained_models_path, gradient_booster_path))


    print('HistGradientBoostingRegressor no MinMax')
    gradient_booster_no_min_max_path = 'best_gradient_booster_model_no_min_max.pkl'
    if TRAIN_MODELS:
        gradient_booster_model = ensemble.HistGradientBoostingRegressor(**best_hist_params)
        best_gradient_booster_model_no_min_max = test_model(gradient_booster_model, x_without_min_max, y)
        if SAVE_MODELS:
            joblib.dump(best_gradient_booster_model_no_min_max, os.path.join(trained_models_path, gradient_booster_no_min_max_path))
    else:
        best_gradient_booster_model_no_min_max = joblib.load(os.path.join(trained_models_path, gradient_booster_no_min_max_path))

    # MLPRegressor
    print('MLPRegressor')
    nn_path = 'best_nn_model.pkl'
    if TRAIN_MODELS:
        nn_model = MLPRegressor(hidden_layer_sizes=[4, 4, 3], max_iter=1000, activation='relu', solver='adam')
        best_nn_model = test_model(nn_model, x, y)
        if SAVE_MODELS:
            joblib.dump(best_nn_model, os.path.join(trained_models_path, nn_path))
    else:
        best_nn_model = joblib.load(os.path.join(trained_models_path, nn_path))

    print('MLPRegressor no MinMax')
    nn_no_min_max_path = 'best_nn_model_no_min_max.pkl'
    if TRAIN_MODELS:
        nn_model = MLPRegressor(hidden_layer_sizes=[4, 4, 3], max_iter=1000, activation='relu', solver='adam')
        best_nn_model_no_min_max = test_model(nn_model, x_without_min_max, y)
        if SAVE_MODELS:
            joblib.dump(best_nn_model_no_min_max, os.path.join(trained_models_path, nn_no_min_max_path))
    else:
        best_nn_model_no_min_max = joblib.load(os.path.join(trained_models_path, nn_no_min_max_path))

    # Plots everything
    if PLOT_MODELS:
        n_features = x_without_min_max[:, 0]
        models = [
            ('LinearRegression d=1 (no Min-Max)', best_linear_model_no_min_max, x_without_min_max),
            ('LinearRegression d=2 (no Min-Max)', best_linear_model_2_no_min_max, x_polynomial_2_without_min_max),
            ('LinearRegression d=3 (no Min-Max)', best_linear_model_3_no_min_max, x_polynomial_3_without_min_max),
            ('HistGradientBoostingRegressor', best_gradient_booster_model, x),
            ('HistGradientBoostingRegressor (no Min-Max)', best_gradient_booster_model_no_min_max, x_without_min_max),
            ('MLPRegressor', best_nn_model, x),
            ('MLPRegressor (no Min-Max)', best_nn_model_no_min_max, x_without_min_max),
        ]

        # Prints prediction for every model
        for model_description, model, data in models:
            _fig, ax = plt.subplots()  # Creates new figure

            # Predicts for all the data
            predictions = model.predict(data)

            execution_means: List[float] = []
            execution_std_errors: List[float] = []
            predicted_execution_means: List[float] = []
            predicted_std_errors: List[float] = []

            # Groups by number_of_features to get the mean and std
            unique_n_features = np.unique(n_features)
            for current_n_features in unique_n_features:
                # Gets index of the current number of features to get the execution/predicted times
                idx = np.where(n_features == current_n_features)

                # Stores real
                current_execution_times = y[idx]
                execution_mean, std_error = get_mean_and_std(current_execution_times)
                execution_means.append(execution_mean)
                execution_std_errors.append(std_error)

                # Stores predictions
                current_predicted_times = predictions[idx]
                predicted_execution_mean, predicted_std_error = get_mean_and_std(current_predicted_times)
                predicted_execution_means.append(predicted_execution_mean)
                predicted_std_errors.append(predicted_std_error)

            # Plots true Y
            ax.errorbar(unique_n_features, execution_means, yerr=execution_std_errors, capsize=4, label='Execution time', marker='o', linewidth=2)
            ax.errorbar(unique_n_features, predicted_execution_means, yerr=predicted_std_errors, capsize=4, label='Predicted execution time', marker='o', linewidth=2)

            plt.legend()
            plt.title(f'{model_name} | Predictions {model_description}')
            plt.xlabel("Number of features")
            plt.ylabel("Time (seconds)")
        plt.show()


def main():
    train_models('clustering')
    train_models('svm')
    train_models('rf')


if __name__ == '__main__':
    main()
