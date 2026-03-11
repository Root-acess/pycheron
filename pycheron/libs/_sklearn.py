"""
pycheron.libs._sklearn — sklearn exposed via pycrn.sklearn.

Usage:
    import pycheron as pycrn
    pycrn.sklearn.metrics.accuracy_score(y, y_pred)
    pycrn.sklearn.model_selection.cross_val_score(...)
    pycrn.sklearn.preprocessing.StandardScaler()
"""

from __future__ import annotations

# Expose key sklearn submodules as attributes
from sklearn import (                           # noqa: F401
    metrics,
    model_selection,
    preprocessing,
    pipeline,
    compose,
    impute,
    feature_selection,
    decomposition,
    linear_model,
    ensemble,
    tree,
    neighbors,
    svm,
    cluster,
    datasets,
)

# Convenience flat imports
from sklearn.metrics import (                   # noqa: F401
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import (           # noqa: F401
    train_test_split, cross_val_score,
    StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV,
)
from sklearn.preprocessing import (            # noqa: F401
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    OneHotEncoder, OrdinalEncoder,
)
from sklearn.pipeline import Pipeline          # noqa: F401
from sklearn.base import BaseEstimator         # noqa: F401
