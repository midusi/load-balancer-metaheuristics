# Optimizing Feature Selection in Breast and Renal cancer through Load Balancing in Apache Spark cluster using Machine Learning techniques

## Introduction

This repository contains the code, data and experimental results of the paper _Optimizing Feature Selection in Breast and Renal cancer through Load Balancing in Apache Spark cluster using Machine Learning techniques_. <!-- TODO: add link when published -->

The replicated metaheuristic is the _Binary Black Hole Algorithm_ in its two versions:
- [Binary black hole algorithm for feature selection and classification on biological data](https://www.sciencedirect.com/science/article/abs/pii/S1568494617301242?via%3Dihub)
- [Improved black hole and multiverse algorithms for discrete sizing optimization of planar structures](https://www.tandfonline.com/doi/full/10.1080/0305215X.2018.1540697)


## Datasets

The datasets used are:

- [Breast Invasive Carcinoma (TCGA, PanCancer Atlas)][breast-dataset]
  - Samples: 1082
  - Genes: 19727
- [Kidney Renal Papillary Cell Carcinoma (TCGA, PanCancer Atlas)][renal-dataset]
  - Samples: 282
  - Genes: 19291
- [Lung Adenocarcinoma (TCGA, PanCancer Atlas)][lung-dataset]
  - Samples: 501
  - Genes: 19293

Ready-to-use files can be downloaded from this folder at [Drive][datasets-drive].
All can be found listed on [the cBioPortal datasets page][cbioportal-datasets]. The required files are `data_clinical_patient.txt` y `data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt`, and should be placed in the respective folders within `Datasets`.


## Models, training data and Load Balancer models

### Survival inference models

The models used for survival forecast inference are as follows:

- **SVM Survival** ([source en Scikit-Surv][svm-surv-source]).
- **RF Survival** ([source en Scikit-Surv][rf-surv-source]).
- **[Clustering][clustering-source] + [Cox Regression][cox-source]**.


### Load balancer training data

To train the Load Balancers for each of the models mentioned above, the data in the `LoadBalancerDatasets` folder is used. There are two types of files:

- Files [ClusteringTimesRecord-2023-08-18.csv](LoadBalancerDatasets%2FClusteringTimesRecord-2023-08-18.csv), 
[RFTimesRecord-2023-08-18.csv](LoadBalancerDatasets%2FRFTimesRecord-2023-08-18.csv), and
[SVMTimesRecord-2023-08-18.csv](LoadBalancerDatasets%2FSVMTimesRecord-2023-08-18.csv): are datasets with data obtained from experiments run in Multiomix on AWS.
- [ClusteringTimesRecord_full-2023-10-24.csv](LoadBalancerDatasets%2FClusteringTimesRecord_full-2023-10-24.csv), and
[SVMTimesRecord_full-2023-10-24.csv](LoadBalancerDatasets%2FSVMTimesRecord_full-2023-10-24.csv): these data were obtained from the experiments performed in the `paper_load_balancing` folder. To obtain the training data from the models, the script `generate_full_and_overfitting_train_file.py` (with parameter `SAVE_FULL_MODEL = True`) must be run. These datasets **also** include the data extracted from Multiomix mentioned above. These datasets are used to demonstrate that if we train the LoadBalancers with more data and more in line with the times obtained from the local Spark cluster (and not from AWS as obtained from Multiomix), the results are better.
- [ClusteringTimesRecord_overfitting-2023-10-24.csv](LoadBalancerDatasets%2FClusteringTimesRecord_overfitting-2023-10-24.csv), and
[SVMTimesRecord_overfitting-2023-10-24.csv](LoadBalancerDatasets%2FSVMTimesRecord_overfitting-2023-10-24.csv): these data were obtained **only** from the experiments performed in the `paper_load_balancing` folder. To obtain the training data from the models, the script `generate_full_and_overfitting_train_file.py` (with parameter `SAVE_FULL_MODEL = False`) must be run. **The data extracted from Multiomix mentioned above are not included.


### Load Balancer trained models

Load Balancer models are located in the `Trained_models` folder. The models are trained with the training data mentioned above. The models are stored in the `Trained_models/<model>` folder where `<model>` is the name of the model for which it was trained.

The training of all these models is performed in the `load_balancer_model.py` script. The models are trained with the parameters obtained in the `grid_search_params.py` script.

If it has the `_full` suffix is because they were trained with the files [ClusteringTimesRecord_full-2023-10-24.csv](LoadBalancerDatasets%2FClusteringTimesRecord_full-2023-10-24.csv), and
[SVMTimesRecord_full-2023-10-24.csv](LoadBalancerDatasets%2FSVMTimesRecord_full-2023-10-24.csv). If it has the `_overfitting` is because they were trained with the files [ClusteringTimesRecord_overfitting-2023-10-24.csv](LoadBalancerDatasets%2FClusteringTimesRecord_overfitting-2023-10-24.csv), y [SVMTimesRecord_overfitting-2023-10-24.csv](LoadBalancerDatasets%2FSVMTimesRecord_overfitting-2023-10-24.csv). 
If they do not have them then the models were trained only with the information extracted from Multiomix in AWS (fewer data).


## Code structure

The main code is in the `core.py` file where the metaheuristic is executed and the results are reported and saved in a CSV file that will go to the `Results` folder. In `main.py` you define the functions and parameters that you pass to `core` to execute.  This way you can run sequential experiments (plain Python) and those developed in a Spark environment in a clean way since both approaches share the pre-processing, the independent runs and the metaheuristic algorithm. In addition, changing the `RUN_TIMES_EXPERIMENT` parameter to `True` runs the code to evaluate how long the execution takes using different amounts of features. In `plot_times.py` is the code to plot these times using the JSON file generated during the experiment.

The `utils.py` file contains import functions, dataset label column binarization, preprocessing, among other useful functions.

The metaheuristics algorithm and its variants can be found in the `metaheuristics.py` file.

The `memory_leak.py` file contains a benchmark with two solutions to the memory leak produced in Spark during function calls inside a mapPartition. Both the problem and the solution are well explained in this [Stack Overflow question][so-memory-leak].

In the `cross_validation_experiments.py` file a series of experiments with the CrossValidation and Survival-SVM models are carried out. This allows to understand a little better how the parameters work. The motivation for this file stems from encountering decreasing fitness for both training and testing as the number of parameters increased (regardless of the kernel used for the SVM model).

In the `grid_search_params.py` file, experiments with GridSearch are carried out to obtain the best parameters of some models that will be trained in the `load_balancer_model.py` file. The results obtained during the execution of this experiment can be found in the file `All models metrics (grid_search).txt`.

In the `load_balancer_model.py` file the training experiments of different models to predict the execution time from the CSV files contained in the `LoadBalancerDatasets` folder are carried out. The parameters used to train the models are those obtained from the experiments carried out in `grid_search_params.py`.

The `generate_full_train_file.py` file contains a script to run through all the experiments performed (found in the `paper_load_balancing` folder) and generate a CSV file with all the training data for the load balancing models. This extracted data is concatenated with the training files found in the `LoadBalancerDatasets` folder.

In the `plot_experiments.py` file is the code that takes the JSON data of the experiments launched from `main.py` to make comparisons between the data of different experiments (with and without load balancer, for example).

In the file `bin_packing.py` you can find a test example of how the Binpacking algorithm works for the equal distribution between Spark workers considering an assumed execution time. And in `binpacking_with_load_balancing.py` is a proof of concept applying the multiple knapsack algorithm to distribute equally the workload among different workers applying random delays during the different iterations.

In the `paper_load_balancing` folder you can also find all the results grouped in an Excel file for each of the experiments. These results consist of:
  - A sheet with the results of execution times. Detailing mean, standard deviation for each execution strategy. And the minimum value among all. In addition, a bar chart is included for visual comparison.
  - A sheet file with the same structure as the previous point but with the idle time results of the cluster workers.
  - A sheet with the predicted time data and the actual time, together with comparative graphs between both times for each of the three predictive models to be evaluated.

The `simulator.py` file contains the development of the Spark load balancing simulator: it simulates the delegation of partitions to a specific node in order to demonstrate the advantages of the PhD race. In the `simulator_data_summary.py` file, it uses all the results of the `simulator.py` to build an overall summary that compares the 3 strategies and defines which one was the best for each of the iterations. These results are stored in the `Results/simulator_results` folder.

To learn more about the SVM Survival or Random Survival Forest models read the  [Scikit-survival blog][scikit-survival-blog].


## Explanation of JSON file fields

A JSON file is generated for each experiment run. The fields and their explanations are listed below:

**NOTE: all values that are arrays are sorted by position. That is, the first element of _number_of_features_, _execution_times_, _predicted_execution_times_, _fitness_, _times_by_iteration_ (among others) are related. Some values are scalar or have another structure.**

- **number_of_features**: number of features used to obtain the fitness value (either using CV with SVM or RF, or Clustering + Cox Regression).
- **execution_times**: time (in seconds) it took to get the fitness value with the _number_of_features_.
- **predicted_execution_times**: _execution_times_ predicted by the load balancer.
- **fitness**: fitness value obtained during the CV process with SVM or RF, or Clustering + Cox Regression.
- **times_by_iteration**: time (in seconds) that each iteration of the SVM took. If no such model is used the value is 0.0. The name may be confusing, but we leave it that way for backward compatibility issues.
- **test_times**: time (in seconds) it took during testing during the CV process with SVM or RF. If Clustering + Cox Regression is used the value is 0.0.
- **train_scores**: fitness values obtained with the training data during the CV process with SVM or RF. If Clustering + Cox Regression is used the value is 0.0.
- **number_of_iterations**: number of SVM iterations it took to converge such a model during the VC process. If no such model is used the value is 0.0. The name may be confusing, but we leave it that way for backward compatibility issues.
- **hosts**: name of the Spark Workers who were in charge of the computation.
- **workers_idle_times**: dictionary with the names of the Workers (_hosts_) as key and the average and standard deviation values of the idle times (in seconds) of that worker in the whole run of the metaheuristic. The idle time of a Worker is calculated as the difference between the sum of the execution times and the time the master waited to obtain the answer (time elapsed between the distribution of data in the cluster and the collect()).
- **workers_idle_times_per_iteration**: dictionary with the names of the Workers (_hosts_) as key. The value of each key is a list with a tuple of two elements: the iteration of the metaheuristic (not to be confused with an independent run) and the idle time (in seconds) of that Worker in that iteration. **Note that here there could be problems if in some iteration a Worker does not receive stars, since neither the host name nor the time it took cannot be obtained, so it would not be included in this dictionary. In an optimal context, where each worker receives at least one star to compute, this dictionary should have for each Worker a list with as many elements as iterations have been configured for the BBHA.**
- **workers_execution_times_per_iteration**: dictionary with the names of the Workers (_hosts_) as key. The value of each key is a list with a tuple of two elements: the iteration of the metaheuristic (not to be confused with an independent run) and the execution time (in seconds) that it took that Worker in that iteration to compute all the assigned stars. **Note that here there could be problems if in some iteration a Worker does not receive stars, since neither the host name nor the time it took cannot be obtained, so it would not be included in this dictionary. In an optimal context, where each worker receives at least one star to compute, this dictionary should have for each Worker a list with as many elements as iterations have been configured for the BBHA.**
- **partition_ids**: Partition ID that was assigned to the star that computed these features.
- **model**: model used. It can be `svm`, `rf`, or `clustering`.
- **dataset**: dataset used. It can be `Breast_Invasive_Carcinoma`, `Kidney_Renal_Papillary_Cell_Carcinoma`, or `Lung_Adenocarcinoma`.
- **parameters**: a string with the parameters of the model used.
- **number_of_samples**: number of patients in the dataset.
- **independent_iteration_time**: time (in seconds) that the entire process took (also known as _Independent Run_).


## Script execution

Spark has problems with importing user-defined modules, so we leave a file called `scripts.zip` which contains all the necessary modules. Now just run the following commands to get everything working:

1. Deploy a Spark cluster using [this repository][big-data-swarm]. You may need to install the dependencies defined in `requirements.txt` in the image deployed in the cluster.
1. Configure everything in the `main.py` with the parameters to execute.
1. Out of the cluster run to be able to use the Python modules: `./zip_modules.sh`.
1. Inside the Cluster Master run: `spark-submit --py-files scripts.zip main.py`.

**NOTE**: if it fails saying that the `spark-logs` folder does not exist, run the command from the master `hdfs dfs -mkdir /spark-logs`.


[scikit-survival-blog]: https://scikit-survival.readthedocs.io/en/stable/user_guide/understanding_predictions.html
[datasets-drive]: https://drive.google.com/drive/folders/1g7DnPkV7MtbLBHrGLWpCnRWV2LRW6v9-?usp=sharing
[breast-dataset]: https://cbioportal-datahub.s3.amazonaws.com/brca_tcga_pan_can_atlas_2018.tar.gz
[lung-dataset]: https://cbioportal-datahub.s3.amazonaws.com/luad_tcga_pan_can_atlas_2018.tar.gz
[renal-dataset]: https://cbioportal-datahub.s3.amazonaws.com/kirp_tcga_pan_can_atlas_2018.tar.gz
[cbioportal-datasets]: https://www.cbioportal.org/datasets
[so-memory-leak]: https://stackoverflow.com/questions/53105508/pyspark-numpy-memory-not-being-released-in-executor-map-partition-function-mem/71700592#71700592
[svm-surv-source]: https://scikit-survival.readthedocs.io/en/stable/user_guide/survival-svm.html
[rf-surv-source]: https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html
[clustering-source]: https://scikit-learn.org/stable/modules/clustering.html
[cox-source]: https://lifelines.readthedocs.io/en/latest/Quickstart.html?highlight=cox%20regression#survival-regression
[big-data-swarm]: https://github.com/jware-solutions/docker-big-data-cluster