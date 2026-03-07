from sklearn.datasets import load_svmlight_file
import numpy as np

def load_data(train_path, test_path):
    """Carica i dataset in formato LIBSVM e restituisce train, test e il dataset completo."""
    from sklearn.datasets import load_svmlight_file
    from sklearn.preprocessing import StandardScaler

    X_train, y_train = load_svmlight_file(train_path)
    X_test, y_test = load_svmlight_file(test_path)

    # Uniforma il numero di feature tra train e test
    X_train, X_test = _align_features(X_train, X_test)

    # Conversione da matrice sparsa a numpy array denso
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # Normalizzazione: fit solo su train, transform su entrambi
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Dataset completo (train + test) per la visualizzazione
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    return X_train, y_train, X_test, y_test, X_all, y_all


def _align_features(X_train, X_test):
    """Allinea il numero di colonne tra train e test (padding con zeri)."""
    from scipy.sparse import hstack, csr_matrix

    n_train = X_train.shape[1]
    n_test = X_test.shape[1]
    if n_train > n_test:
        diff = n_train - n_test
        X_test = hstack([X_test, csr_matrix((X_test.shape[0], diff))])
    elif n_test > n_train:
        diff = n_test - n_train
        X_train = hstack([X_train, csr_matrix((X_train.shape[0], diff))])
    return X_train, X_test