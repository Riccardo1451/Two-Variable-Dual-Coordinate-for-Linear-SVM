from sklearn.datasets import load_svmlight_file
import numpy as np


def load_data(train_path, test_path, use_scaling=True):
    """Carica i dataset in formato LIBSVM e restituisce train, test e il dataset completo.

    Args:
        train_path: percorso del file train in formato LIBSVM.
        test_path: percorso del file test in formato LIBSVM.
        use_scaling: se True applica StandardScaler fit su train e transform su test.
    """
    from sklearn.datasets import load_svmlight_file
    from sklearn.preprocessing import StandardScaler

    

    X_train, y_train = load_svmlight_file(train_path)
    
    if test_path is not None:
        X_test, y_test = load_svmlight_file(test_path)

        # Uniforma il numero di feature tra train e test
        X_train, X_test = _align_features(X_train, X_test)

    # Conversione da matrice sparsa a numpy array denso
    X_train = X_train.toarray()
    if test_path is not None:
        X_test = X_test.toarray()

    if use_scaling:
        # Normalizzazione: fit solo su train, transform su entrambi
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


X, y = load_svmlight_file("dataset/w8a_t.txt")

print(f"Istanze: {X.shape[0]}, Feature: {X.shape[1]}")
print(f"Nonzero totali: {X.nnz}, Sparsity: {X.nnz / (X.shape[0] * X.shape[1]):.4f}, Media per campione: {X.nnz / X.shape[0]:.2f}")