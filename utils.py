import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

def load_data(train_path, test_path, use_scaling=True):
    """
    Load completely the data from given paths, from LIBSVM, train test and all together.
    """

    X_train, y_train = load_svmlight_file(train_path)
    
    if test_path is not None:
        X_test, y_test = load_svmlight_file(test_path)

        # Uniform number of features between train and test (padding with zeros)
        X_train, X_test = _align_features(X_train, X_test)

    # Convert from sparse matrix to array
    X_train = X_train.toarray()
    if test_path is not None:
        X_test = X_test.toarray()

    if use_scaling:
        # Normalization: fit on train, transform on both
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if test_path is not None:
            X_test = scaler.transform(X_test)

    if test_path is None:
        X_test, y_test = None, None
        X_all = X_train.copy()
        y_all = y_train.copy()
    else:
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])

    return X_train, y_train, X_test, y_test, X_all, y_all


def _align_features(X_train, X_test):
    """Uniform number of features between train and test (padding with zeros)."""

    n_train = X_train.shape[1]
    n_test = X_test.shape[1]
    if n_train > n_test:
        diff = n_train - n_test
        X_test = hstack([X_test, csr_matrix((X_test.shape[0], diff))])
    elif n_test > n_train:
        diff = n_test - n_train
        X_train = hstack([X_train, csr_matrix((X_train.shape[0], diff))])
    return X_train, X_test