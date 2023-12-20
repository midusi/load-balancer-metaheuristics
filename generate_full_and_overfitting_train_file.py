# Generates the clustering_training_experiments.csv and svm_training_experiments.csv from the data extracted from the
# experiments run in the cluster.
import json
import os
import numpy as np
import pandas as pd

FOLDER = '../paper_load_balancing'

CLUSTERING_FOLDER = 'clustering'
CLUSTERING_EXISTING_MODEL = '../LoadBalancerDatasets/ClusteringTimesRecord-2023-08-18.csv'
CLUSTERING_NEW_MODEL_FULL = '../LoadBalancerDatasets/ClusteringTimesRecord_full-2023-10-24.csv'
CLUSTERING_NEW_MODEL_OVERFITTING = '../LoadBalancerDatasets/ClusteringTimesRecord_overfitting-2023-10-24.csv'

SVM_FOLDER = 'svm'
SVM_EXISTING_MODEL = '../LoadBalancerDatasets/SVMTimesRecord-2023-08-18.csv'
SVM_NEW_MODEL_FULL = '../LoadBalancerDatasets/SVMTimesRecord_full-2023-10-24.csv'
SVM_NEW_MODEL_OVERFITTING = '../LoadBalancerDatasets/SVMTimesRecord_overfitting-2023-10-24.csv'

# If this variable is True saves the full model, otherwise saves the overfitting model
SAVE_FULL_MODEL = False

# To show values count and other extra data about final DataFrame
DEBUG = False


def clustering_training():
    clustering_path = os.path.join(FOLDER, CLUSTERING_FOLDER)
    print(f'Clustering training. Getting data from {clustering_path}')

    # Iterates over all the JSON files in the clustering folder (recursively)
    data = {
        'Number of features': [],
        'Number of samples': [],
        'Execution time': [],
        'Fitness': [],
        'Train score': [],
        'Number of clusters': [],
        'Algorithm': [],
        'Scoring method': []
    }

    for (root, dirs, files) in os.walk(clustering_path, topdown=False):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                    number_of_samples = json_data['number_of_samples']
                    n_elements = len(json_data['number_of_features'])
                    # Generates an array with the number of samples repeated n_elements times
                    number_of_samples = [number_of_samples] * n_elements
                    number_of_clusters = 2  # Fixed value for the clustering experiments
                    algorithm = 1  # K_MEANS enum in Multiomix
                    scoring_method = 2  # LOG_LIKELIHOOD enum in Multiomix

                    data['Number of features'].extend(json_data['number_of_features'])
                    data['Number of samples'].extend(number_of_samples)
                    data['Execution time'].extend(json_data['execution_times'])
                    data['Fitness'].extend(json_data['fitness'])
                    data['Train score'].extend(json_data['train_scores'])
                    data['Number of clusters'].extend([number_of_clusters] * n_elements)
                    data['Algorithm'].extend([algorithm] * n_elements)
                    data['Scoring method'].extend([scoring_method] * n_elements)

    # Creates DataFrame
    df = pd.DataFrame(data)

    print(f'Shape of new model: {df.shape}')

    # Adds the data from the existing model
    if SAVE_FULL_MODEL:
        existing_model_df = pd.read_csv(CLUSTERING_EXISTING_MODEL)
        print(f'Shape of existing model: {existing_model_df.shape}')
        df = pd.concat([df, existing_model_df], ignore_index=True)

    print(f'Shape of DF after concat: {df.shape}')

    # Removes rows with NaNs values, infinite values or the column 'Execution time' with value -1
    print('Removing NaNs and infinite values...')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df[df['Execution time'] != -1]

    print(df.head())
    print(f'Shape of final DF: {df.shape}')

    # Counts different values for each column
    if DEBUG:
        print(df['Number of features'].value_counts())
        print(df['Number of samples'].value_counts())
        print(df['Number of clusters'].value_counts())
        print(df['Algorithm'].value_counts())
        print(df['Scoring method'].value_counts())

    # Saves the DataFrame to a CSV file
    final_path = CLUSTERING_NEW_MODEL_FULL if SAVE_FULL_MODEL else CLUSTERING_NEW_MODEL_OVERFITTING
    if not os.path.exists(final_path):
        df.to_csv(final_path, index=False)
    else:
        print(f'Clustering model path "{final_path}" already exists. Omitting...')


def __get_kernel_int_value(value: str) -> int:
    """Returns the corresponding integer value for the kernel string value (Multiomix equivalent)."""
    if value == 'linear':
        return 1
    if value == 'poly':
        return 2
    if value == 'rbf':
        return 3

    raise ValueError(f'Invalid kernel value: {value}')


def svm_training():
    svm_path = os.path.join(FOLDER, SVM_FOLDER)
    print(f'SVM training. Getting data from {svm_path}')

    # Iterates over all the JSON files in the svm folder (recursively)
    data = {
        'Number of features': [],
        'Number of samples': [],
        'Execution time': [],
        'Fitness': [],
        'Train score': [],
        'Test time': [],
        'Number of iterations': [],
        'Time by iteration': [],
        'Max iterations': [],
        'Optimizer': [],
        'Kernel': []
    }

    for (root, dirs, files) in os.walk(svm_path, topdown=False):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                    number_of_samples = json_data['number_of_samples']
                    n_elements = len(json_data['number_of_features'])

                    # Generates an array with the number of samples repeated n_elements times
                    number_of_samples = [number_of_samples] * n_elements
                    max_iterations = 1000  # Fixed value for the SVM experiments
                    optimizer = 'avltree'  # Fixed value for the SVM experiments

                    # The kernel is the second to last after splitting the 'parameters' string by '_'
                    kernel = json_data['parameters'].split('_')[-2]
                    kernel = __get_kernel_int_value(kernel)

                    data['Number of features'].extend(json_data['number_of_features'])
                    data['Number of samples'].extend(number_of_samples)
                    data['Execution time'].extend(json_data['execution_times'])
                    data['Fitness'].extend(json_data['fitness'])
                    data['Train score'].extend(json_data['train_scores'])
                    data['Test time'].extend(json_data['test_times'])
                    data['Number of iterations'].extend(json_data['number_of_iterations'])
                    data['Time by iteration'].extend(json_data['times_by_iteration'])
                    data['Max iterations'].extend([max_iterations] * n_elements)
                    data['Optimizer'].extend([optimizer] * n_elements)
                    data['Kernel'].extend([kernel] * n_elements)

    # Creates DataFrame
    df = pd.DataFrame(data)

    print(f'Shape of new model: {df.shape}')

    # Adds the data from the existing model
    if SAVE_FULL_MODEL:
        existing_model_df = pd.read_csv(SVM_EXISTING_MODEL)
        print(f'Shape of existing model: {existing_model_df.shape}')
        df = pd.concat([df, existing_model_df], ignore_index=True)

    print(f'Shape of DF after concat: {df.shape}')

    # Removes rows with NaNs values, infinite values or the column 'Execution time' with value -1
    print('Removing NaNs and infinite values...')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df[df['Execution time'] != -1]

    print(df.head())
    print(f'Shape of final DF: {df.shape}')

    # Counts different values for each column
    if DEBUG:
        print(df['Number of features'].value_counts())
        print(df['Number of samples'].value_counts())
        print(df['Max iterations'].value_counts())
        print(df['Optimizer'].value_counts())
        print(df['Kernel'].value_counts())

    # Saves the DataFrame to a CSV file
    final_path = SVM_NEW_MODEL_FULL if SAVE_FULL_MODEL else SVM_NEW_MODEL_OVERFITTING
    if not os.path.exists(final_path):
        df.to_csv(final_path, index=False)
    else:
        print(f'SVM model path "{final_path}" already exists. Omitting...')


if __name__ == '__main__':
    clustering_training()
    svm_training()
