import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.5449149683919614
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        make_pipeline(
            PCA(iterated_power=7, svd_solver="randomized"),
            RobustScaler()
        )
    ),
    LGBMClassifier(early_stopping_rounds=None, min_child_samples=100, min_child_weight=0.001, n_estimators=100, num_leaves=10, reg_alpha=1, reg_lambda=50)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
