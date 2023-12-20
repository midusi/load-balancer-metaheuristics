# Paper

## Introducción

Lo desarrollado acá buscan implementar los algoritmos definidos en los siguientes papers:
- [Binary black hole algorithm for feature selection and classification on biological data](https://www.sciencedirect.com/science/article/abs/pii/S1568494617301242?via%3Dihub): la metaheurística en sí misma. 
- [Improved black hole and multiverse algorithms for discrete sizing optimization of planar structures](https://www.tandfonline.com/doi/full/10.1080/0305215X.2018.1540697): el paper que tunea un poco al algoritmo.

**Papers que podrían ser útiles:**

- [A Parallel Random Forest Algorithm for Big Data in a Spark Cloud Computing Environment](https://arxiv.org/abs/1810.07748): implementación de un Random Forest en Spark. **Útil ya que el Random Forest que viene con Spark parece ser lentísimo.**
- [A Multi Dynamic Binary Black Hole Algorithm Applied to Set Covering Problem](https://link.springer.com/chapter/10.1007%2F978-981-10-3728-3_6): implementación de una variante del algoritmo en Spark. **IMPORTANTE:** lo que tiene este paper es que distribuye el algoritmo de BH pero no los modelos. Es decir, distribuyen lo menos costoso cuando nosotros buscamos paralelizar la parte de los modelos que arrojan la métrica a optimizar. Por ende, este paper no nos sería muy útil, quizás para un cita en la sección del estado del arte.


## Instalación 

Para correr el código hay que instalar las dependencias:

1. Crear un virtual env: `python3 -m venv venv` (una única vez).
2. Activar el virtual env: `source venv/bin/activate` (solo cuando se utilice)
    1. Para salir del virtual env ejecutar: `deactivate`
3. Instalar las dependencias: `pip install -r requirements.txt`


## Datasets

Los datasets utilizados son:

- [Breast Invasive Carcinoma (TCGA, PanCancer Atlas)][breast-dataset]
  - Samples: 1082
  - Genes: 19727
  - Objetivo: entrenamiento de tiempos
- [Kidney Renal Papillary Cell Carcinoma (TCGA, PanCancer Atlas)][renal-dataset]
  - Samples: 282
  - Genes: 19291
  - Objetivo: entrenamiento de tiempos
- [Lung Adenocarcinoma (TCGA, PanCancer Atlas)][lung-dataset]
  - Samples: 501
  - Genes: 19293
  - Objetivo: utilizar como datos de testing en pruebas reales con los modelos ya entrenados por los otros datasets de entrenamiento de tiempos

Se pueden descargar los archivos ya listos para utilizar desde esta carpeta en [Drive][datasets-drive]
Todos pueden encontrarse listados en [la página de datasets de cBioPortal][cbioportal-datasets]). Los archivos requeridos son `data_clinical_patient.txt` y `data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt`, y deben ponerse en las respectivas carpetas dentro de `Datasets`.


## Modelos, datos de entrenamiento y Load Balancer


### Modelos

Los modelos utilizados son los siguientes:

- **SVM Survival** ([source en Scikit-Surv][svm-surv-source]): se utiliza para tareas de Ranking y Regression.
- **RF Survival** ([source en Scikit-Surv][rf-surv-source]): se utiliza para tareas de Regression. Pero anda lentísimo
- **Clustering + Cox Regression**: se utilizan diferentes [algoritmos de clustering][group-source] para agrupar por expresión y después se obtiene el C-Index/Log-Likelihood a través de [Cox Regression][cox-source] para evaluar que tan diferentes son los grupos de pacientes.


### Datos de entrenamiento de Load balancer

Para entrenar a los Load Balancers para cada uno de los modelos mencionados arriba se utilizan los datos de la carpeta `LoadBalancerDatasets`. Hay dos tipos de archivos:

- Los archivos [ClusteringTimesRecord-2023-08-18.csv](LoadBalancerDatasets%2FClusteringTimesRecord-2023-08-18.csv), 
[RFTimesRecord-2023-08-18.csv](LoadBalancerDatasets%2FRFTimesRecord-2023-08-18.csv), y
[SVMTimesRecord-2023-08-18.csv](LoadBalancerDatasets%2FSVMTimesRecord-2023-08-18.csv): son datasets con datos que se obtuvieron de experimentos ejecutados en Multiomix.
- [ClusteringTimesRecord_full-2023-10-24.csv](LoadBalancerDatasets%2FClusteringTimesRecord_full-2023-10-24.csv), y
[SVMTimesRecord_full-2023-10-24.csv](LoadBalancerDatasets%2FSVMTimesRecord_full-2023-10-24.csv): estos datos fueron obtenidos a partir de los experimentos realizados en la carpeta `paper_load_balancing`. Para obtener los datos de entrenamiento de los modelos se debe ejecutar el script `generate_full_and_overfitting_train_file.py` (con parámetro `SAVE_FULL_MODEL = True`). Estos datasets **también** incluyen los datos extraídos de Multiomix mencionados arriba. Estos conjuntos se utilizan para demostrar que si entrenamos a los LoadBalancers con más datos y más acordes a los tiempos obtenidos en el cluster local de Spark (y no de AWS como los que se obtuvieron desde Multiomix), los resultados son mejores.
- [ClusteringTimesRecord_overfitting-2023-10-24.csv](LoadBalancerDatasets%2FClusteringTimesRecord_overfitting-2023-10-24.csv), y
[SVMTimesRecord_overfitting-2023-10-24.csv](LoadBalancerDatasets%2FSVMTimesRecord_overfitting-2023-10-24.csv): estos datos fueron obtenidos **únicamente** a partir de los experimentos realizados en la carpeta `paper_load_balancing`. Para obtener los datos de entrenamiento de los modelos se debe ejecutar el script `generate_full_and_overfitting_train_file.py` (con parámetro `SAVE_FULL_MODEL = False`). **No incluyen los datos extraídos de Multiomix mencionados arriba.**


### Modelos de Load Balancer

Los modelos de Load Balancer se encuentran en la carpeta `Trained_models`. Los modelos se entrenan con los datos de entrenamiento mencionados arriba. Los modelos se guardan en la carpeta `Trained_models/<model>` donde `<model>` es el nombre del modelo para el cual fue entrenado.

El entrenamiento de todos estos modelos se realizan en el script `load_balancer_model.py`. Los modelos se entrenan con los parámetros obtenidos en el script `grid_search_params.py`.

Si tienen el sufijo `_full` es porque fueron entrenados con los datos [ClusteringTimesRecord_full-2023-10-24.csv](LoadBalancerDatasets%2FClusteringTimesRecord_full-2023-10-24.csv), y
[SVMTimesRecord_full-2023-10-24.csv](LoadBalancerDatasets%2FSVMTimesRecord_full-2023-10-24.csv). Si tiene el sufijo `_overfitting` es porque fueron entrenados con los datos [ClusteringTimesRecord_overfitting-2023-10-24.csv](LoadBalancerDatasets%2FClusteringTimesRecord_overfitting-2023-10-24.csv), y [SVMTimesRecord_overfitting-2023-10-24.csv](LoadBalancerDatasets%2FSVMTimesRecord_overfitting-2023-10-24.csv). 
Si no los tienen entonces los modelos fueron entrenados solo con la información extraída de Multiomix (menos datos).


## Organización del código

El código principal está en el archivo `core.py` donde se ejecuta la metaheurística y se informan y guardan los resultados en un archivo CSV que irá a parar a la carpeta `Results`. En `main.py` se definen las funciones y parámetros que se pasa al `core` para ejecutar.  De esta manera se pueden correr los experimetos secuenciales (Python plano) y los desarrollados en un entorno Spark de manera limpia ya que ambos enfoques comparten el pre-procesamiento, las corridas independientes y el algoritmo de la metaheurística. Además, cambiando el parámetro `RUN_TIMES_EXPERIMENT` a `True` se ejecuta el código para evaluar cuánto tarda la ejecución utilizando diferentes cantidades de features. En `plot_times.py` se encuentra el código para graficar dichos tiempos utilizando el archivo JSON generado durante el experimento.

En el archivo `utils.py` se encuentran las funciones de importación, de  binarización de la columna del label del dataset, de preprocesamiento, entre otras funciones útiles.

El algoritmo de la metaheurística y sus variantes se pueden encontrar en el archivo `metaheuristics.py`.

El archivo `memory_leak.py` contiene un benchmark con dos soluciones al memory leak producido en Spark durante el llamado a funciones dentro de un mapPartition. Tanto el problema como la solución están bien explicados en esta pregunta de [Stack Overflow][so-memory-leak].

En el archivo `cross_validation_experiments.py` se lleva a cabo una serie de experimentos con el CrossValidation y los modelos Survival-SVM. Esto permite entender un poco mejor el funcionamiento de los parámetros. La motivación de este archivo nace de encontrarse con un fitness decreciente tanto para training como para testing a medida que el número de parámetros aumentaba (independientemente del kernel que se utilizaba para el modelo SVM).

En el archivo `grid_search_params.py` se lleva a cabo experimentos con GridSearch para obtener los mejores parámetros de algunos modelos que serán entrenados en el archivo `load_balancer_model.py`. Los resultados obtenidos durante la ejecución de este experimento se encuentran en el archivo `All models metrics (grid_search).txt`.

En el archivo `load_balancer_model.py` se llevan a cabo los experimentos de entrenamiento de diferentes modelos para predecir el tiempo de ejecucion a partir de los archivos CSV contenidos en la carpeta `LoadBalancerDatasets`. Los parámetros utilizados para entrenar los modelos son los obtenidos a partir de los experimentos llevados a cabo en `grid_search_params.py`.

El archivo `generate_full_train_file.py` contiene un script para recorrer todos los experimentos realizados (que se encuentran en la carpeta `paper_load_balancing`) y generar un archivo CSV con todos los datos de entrenamiento para los modelos de load balancing. Estos datos extraídos se concatenan con los archivos de entrenamiento que se encuentran en la carpeta `LoadBalancerDatasets`.

En el archivo `plot_experiments.py` está el código que toma los datos en JSON de los experimentos lanzados desde `main.py` para realizar comparaciones entre los datos de diferentes experimentos (con y sin load balancer, por ejemplo).

En el archivo `bin_packing.py` se encuentra un ejemplo de prueba de como funciona el algoritmo Binpacking para la distribución equitativa entre workers de Spark considerando un supuesto tiempo de ejecución. Y en `binpacking_with_load_balancing.py` se realiza una prueba de concepto aplicando el algoritmo multiple knapsack para distribuir equitativamente la carga de trabajo entre diferentes workers aplicando delays aleatorios durente las diferentes iteraciones.

En el archivo `simulator.py` se encuentra el desarrollo del simulador de balanceo de carga de Spark: simula la delegación de las particiones a un nodo específico para poder demostrar las ventajas de la carrera de doctorado. En el archivo `simulator_data_summary.py` utiliza todos los resultados del `simulator.py` para armar un summary general que compare las 3 estrategias y defina cuál fue la mejor para cada una de las iteraciones.

Para conocer más sobre los modelos SVM Survival o Random Survival Forest leer el blog de [Scikit-survival][scikit-survival-blog].


## Explicación de los campos de los archivos JSON

En cada corrida de experimentos se genera un archivo JSON. A continuación se listan los campos con sus explicaciones:

**NOTA: todos los valores que son arreglos se encuentran ordenados por posición. Es decir, el primer elemento de _number_of_features_, _execution_times_, _predicted_execution_times_, _fitness_, _times_by_iteration_ (entre otros) están relacionados. Algunos valores son escalares o tienen otra estructura.**

- **number_of_features**: número de features utilizados para obtener el valor de fitness (ya sea utilizando CV con SVM o RF, o Clustering + Cox Regression).
- **execution_times**: tiempo (en segundos) que tomó obtener el valor de fitness con el _number_of_features_.
- **predicted_execution_times**: _execution_times_ predicho por el load balancer. 
- **fitness**: valor de fitness obtenido durante el proceso de CV con SVM o RF, o Clustering + Cox Regression.
- **times_by_iteration**: tiempo (en segundos) que llevó cada iteración del SVM. Si no se utiliza dicho modelo el valor es 0.0. El nombre puede ser confuso pero lo dejamos así por temas de retrocompatibilidad.
- **test_times**: tiempo (en segundos) que llevó durante el testing durante el proceso de CV con SVM o RF. Si se utiliza Clustering + Cox Regression el valor es 0.0.
- **train_scores**: valores de fitness obtenidos con los datos de entrenamiento durante el proceso de CV con SVM o RF. Si se utiliza Clustering + Cox Regression el valor es 0.0. 
- **number_of_iterations**: número de iteraciones de SVM que llevó para converger dicho modelo durante el proceso de CV. Si no se utiliza dicho modelo el valor es 0.0. El nombre puede ser confuso pero lo dejamos así por temas de retrocompatibilidad.
- **hosts**: nombre de los Workers de Spark que fueron los encargados de computar.
- **workers_idle_times**: diccionario con los nombres de los Workers (_hosts_) como clave y los valores de promedio y desvío estandar de los tiempos ociosos (en segundos) de dicho worker en toda la corrida de la metaheurística. El tiempo ocioso de un Worker se calcula como la diferencia entre la suma de los tiempos de ejecución y lo que esperó el master en obtener la respuesta (tiempo transcurrido entre la distribución de datos en el cluster hasta hacer el collect()).
- **workers_idle_times_per_iteration**: diccionario con los nombres de los Workers (_hosts_) como clave. El valor de cada clave es una lista con una tupla de dos elementos: la iteración de la metaheurística (no confundir con una corrida independiente) y el tiempo ocioso (en segundos) de ese Worker en dicha iteración. **Nótese que acá podría haber problemas si en alguna iteración un Worker no recibe estrellas, ya que ahí no se puede obtener ni el nombre del host, ni el tiempo que demoró, por lo que no sería incluído en este diccionario. En un contexto óptimo, donde cada worker recibe al menos una estrella para computar, este diccionario debería tener por cada Worker una lista con tantos elementos como iteraciones se hayan configurado para el BBHA.**
- **workers_execution_times_per_iteration**: diccionario con los nombres de los Workers (_hosts_) como clave. El valor de cada clave es una lista con una tupla de dos elementos: la iteración de la metaheurística (no confundir con una corrida independiente) y el tiempo de ejecución (en segundos) que le llevó a ese Worker en dicha iteración computar todas las estrellas asignadas. **Nótese que acá podría haber problemas si en alguna iteración un Worker no recibe estrellas, ya que ahí no se puede obtener ni el nombre del host, ni el tiempo que demoró, por lo que no sería incluído en este diccionario. En un contexto óptimo, donde cada worker recibe al menos una estrella para computar, este diccionario debería tener por cada Worker una lista con tantos elementos como iteraciones se hayan configurado para el BBHA.**
- **partition_ids**: ID de partición que se asignó a la estrella que computó estos features.
- **model**: modelo utilizado. Puede ser `svm`, `rf`, `clustering`. 
- **dataset**: dataset utilizado. Puede ser `Breast_Invasive_Carcinoma`, `Kidney_Renal_Papillary_Cell_Carcinoma`, `Lung_Adenocarcinoma`.
- **parameters**: un string con los parámetros del modelo utilizado.
- **number_of_samples**: número de pacientes que contiene el dataset.
- **independent_iteration_time**: tiempo (en segundos) que llevó el proceso completo (también conocido como _Corrida independiente_).


## Ejecución de scripts

Spark tiene problemas con la importación de modulos definidos por el usuario, por lo que se deja un archivo llamado `scripts.zip` que contiene todos los modulos necesarios. Ahora solo hay que correr los siguientes comandos para que funcione todo:

1. Configurar todo en el `main.py` con los parámetros a ejecutar.
2. Estando fuera del cluster ejecutar para poder utilizar los modulos de Python: `./zip_modules.sh`
3. Estando dentro del Master del cluster: `spark-submit --py-files scripts.zip main.py`

Si falla diciendo que no existe la carpeta `spark-logs` hay que correr el comando desde el master `hdfs dfs -mkdir /spark-logs`.


## Como manejar los resultados

Para organizar todos los resultados se realizan los siguientes pasos:

1. Correr el script direccionando tanto el stderr como el stdout a un archivo. Por ejemplo: `spark-submit --py-files scripts.zip main.py &> logs_my_script.txt`. Cuando termine tanto el script secuencial como el de Spark, quedará en la carpeta `Results` un `.csv` con el datetime exacto en el que se corrió el script con todos los resultados obtenidos.
2. Poner el `logs_my_script.txt` junto con un el `.csv` generado en la carpeta `Results/results_to_push/[datetime]/` (se debe crear manualmente esa carpeta).
    1. Opcional: renombrar el `.txt` para que también tenga el datetime. De esa forma deberían quedar dos archivos `[datetime].csv` y `[datetime].txt` en la carpeta `[datetime]`
3. Crear un `README.txt` que contenga información sobre el experimento, como:
    - Si es secuencial o se lanzó en un cluster Spark.
    - Si se usaron parámetros personalizados o se dejó todo por defecto.
    - Cualquier otro dato que se considere útil para el paper.
4. Pushear ambos archivos!


[scikit-survival-blog]: https://scikit-survival.readthedocs.io/en/stable/user_guide/understanding_predictions.html
[datasets-drive]: https://drive.google.com/drive/folders/1g7DnPkV7MtbLBHrGLWpCnRWV2LRW6v9-?usp=sharing
[breast-dataset]: https://cbioportal-datahub.s3.amazonaws.com/brca_tcga_pan_can_atlas_2018.tar.gz
[lung-dataset]: https://cbioportal-datahub.s3.amazonaws.com/luad_tcga_pan_can_atlas_2018.tar.gz
[renal-dataset]: https://cbioportal-datahub.s3.amazonaws.com/kirp_tcga_pan_can_atlas_2018.tar.gz
[cbioportal-datasets]: https://www.cbioportal.org/datasets
[so-memory-leak]: https://stackoverflow.com/questions/53105508/pyspark-numpy-memory-not-being-released-in-executor-map-partition-function-mem/71700592#71700592
[svm-surv-source]: https://scikit-survival.readthedocs.io/en/stable/user_guide/survival-svm.html
[rf-surv-source]: https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html
[group-source]: https://scikit-learn.org/stable/modules/clustering.html
[cox-source]: https://lifelines.readthedocs.io/en/latest/Quickstart.html?highlight=cox%20regression#survival-regression