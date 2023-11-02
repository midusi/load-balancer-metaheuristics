# Este script permite probar que pasa con los datos NaNs cuando se entrena el modelo de scikit-surv
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import cross_val_score
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM

df = pd.read_json('bug_nans/data.json', orient='split')
with open('bug_nans/y.json', 'r') as file:
    y = json.loads(file.read())
    # y = np.array(y, dtype=(bool, float))
    # y = np.asarray(y)
    y = np.array([np.array(xi, dtype='object') for xi in y])
    y = np.core.records.fromarrays(y.transpose(), names='event, time', formats='bool, float')
    print(y)

# print(y.shape)

# print(df.head())
# print(df.shape)
# print(np.isinf(df.values).any())
# print(np.isnan(df.values).any())
# print(np.isfinite(df.values).all())
#
#
# print(df[df.eq(-3.09700602e+005).any(1)])
# exit()


RANDOM_STATE = 20
# CLASSIFIER = FastSurvivalSVM(rank_ratio=0.0, max_iter=1000, tol=1e-5, random_state=RANDOM_STATE)
CLASSIFIER = FastKernelSurvivalSVM(rank_ratio=0.0, max_iter=1000, tol=1e-5, random_state=RANDOM_STATE)

res = cross_val_score(
        CLASSIFIER,
        df.values,
        y,
        cv=10,
        n_jobs=-1
    )
concordance_index_mean = res.mean()
print(f'concordance_index_mean -> {concordance_index_mean}')
