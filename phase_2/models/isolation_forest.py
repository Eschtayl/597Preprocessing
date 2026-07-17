import numpy as np
from sklearn.ensemble import IsolationForest

from config import (
    RANDOM_SEED,
    IF_N_ESTIMATORS,
    IF_MAX_SAMPLES,
    IF_CONTAMINATION,
    IF_N_JOBS,
)


def train_isolation_forest(x_train, y_train=None, seed=RANDOM_SEED):
    # Fit on benign training rows only (normality model; labels filter train set, not used as targets)
    # Accepts a DataFrame (raw features) or a numpy array (e.g. AE latent embedding)
    x_arr = x_train.to_numpy() if hasattr(x_train, 'to_numpy') else np.asarray(x_train)
    if y_train is not None:
        benign_mask = np.asarray(y_train) == 0
        x_fit = x_arr[benign_mask]
    else:
        x_fit = x_arr

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=IF_MAX_SAMPLES,
        contamination=IF_CONTAMINATION,
        random_state=seed,
        n_jobs=IF_N_JOBS,
    )
    model.fit(x_fit)
    print(f'Isolation Forest fitted on {len(x_fit)} benign training rows')
    return model


def get_anomaly_scores(model, x_data):
    # Higher score = more anomalous (negated sklearn decision_function)
    x_arr = x_data.to_numpy() if hasattr(x_data, 'to_numpy') else np.asarray(x_data)
    return -model.decision_function(x_arr)
