import numpy as np 
from utils import load_data
from tqdm import tqdm
import time

class SVM_2CD:
    def __init__(self, C=1.0, n_iters=1000, tol=1e-6):
        self.C = C
        self.n_iters = n_iters
        self.tol = tol

        self.alpha = None
        self.w = None
        self.fobj_history = []
        self.fobj_history_cd = []

    def _solve_2d_subproblem(self, ai, aj, Gi, Gj, Qii, Qjj, Qij):
        delta = Qii * Qjj - Qij ** 2

        # Se il sistema 2x2 e' mal condizionato, usa due update 1D separati.
        if delta <= 1e-8:
            di = 0.0
            dj = 0.0

            if not (ai == 0 and Gi >= 0):
                ai_new = max(0.0, ai - Gi / Qii)
                di = ai_new - ai

            if not (aj == 0 and Gj >= 0):
                aj_new = max(0.0, aj - Gj / Qjj)
                dj = aj_new - aj

            return di, dj

        # Soluzione libera del sottoproblema 2D.
        di = (-Qjj * Gi + Qij * Gj) / delta
        dj = (-Qii * Gj + Qij * Gi) / delta

        ai_new = ai + di
        aj_new = aj + dj

        # proiezione
        ai_new = max(0.0, ai_new)
        aj_new = max(0.0, aj_new)

        return ai_new - ai, aj_new - aj

    def fit(self, X, y):
        # Bias embedding (no vincolo duale)
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        y = np.where(y <= 0, -1, 1).astype(float)

        n_samples, n_features = X.shape

        self.alpha = np.zeros(n_samples)
        w = np.zeros(n_features)

        # L2-SVM
        Dii = 1.0 / (2.0 * self.C)
        Q_diag = np.sum(X**2, axis=1) + Dii

        self.fobj_history = []
        self.fobj_history_cd = []
        total_step = 0
        total_cd_step = 0
        start = time.time()

        LOG_INTERVAL = max(1, n_samples // 10)

        for epoch in tqdm(range(self.n_iters), desc="Epoche", unit="epoch"):
            M = -np.inf
            m = np.inf

            perm = np.random.permutation(n_samples)

            # schema (π(1),π(2)), (π(3),π(4)), ...
            for k in range(0, n_samples - 1, 2):
                i, j = perm[k], perm[k + 1]
                total_cd_step += 2

                # gradienti
                Gi = y[i] * np.dot(w, X[i]) - 1 + Dii * self.alpha[i]
                Gj = y[j] * np.dot(w, X[j]) - 1 + Dii * self.alpha[j]

                # projected gradients
                PGi = min(0.0, Gi) if self.alpha[i] == 0 else Gi
                PGj = min(0.0, Gj) if self.alpha[j] == 0 else Gj

                M = max(M, PGi, PGj)
                m = min(m, PGi, PGj)

                # skipping
                if (self.alpha[i] == 0 and Gi >= 0) and (self.alpha[j] == 0 and Gj >= 0):
                    continue

                Qii = Q_diag[i]
                Qjj = Q_diag[j]
                Qij = y[i] * y[j] * np.dot(X[i], X[j])

                di, dj = self._solve_2d_subproblem(
                    self.alpha[i], self.alpha[j],
                    Gi, Gj, Qii, Qjj, Qij
                )

                if abs(di) > 1e-12 or abs(dj) > 1e-12:
                    w += di * y[i] * X[i] + dj * y[j] * X[j]
                    self.alpha[i] += di
                    self.alpha[j] += dj

                total_step += 1

                # logging
                if total_step % LOG_INTERVAL == 0:
                    fobj = (
                        0.5 * np.dot(w, w)
                        - np.sum(self.alpha)
                        + 0.5 * Dii * np.dot(self.alpha, self.alpha)
                    )
                    self.fobj_history.append(
                        (time.time() - start, total_step, fobj)
                    )

                # Logging con asse x basato sui passi CD tentati.
                if total_cd_step % LOG_INTERVAL == 0:
                    fobj = (
                        0.5 * np.dot(w, w)
                        - np.sum(self.alpha)
                        + 0.5 * Dii * np.dot(self.alpha, self.alpha)
                    )
                    self.fobj_history_cd.append(
                        (time.time() - start, total_cd_step, fobj)
                    )
                    

            # criterio di stop IDENTICO al 1-CD
            if M - m < self.tol:
                print(f"\nConvergenza raggiunta all'epoca {epoch}")
                break

        final_obj = (
            0.5 * np.dot(w, w)
            - np.sum(self.alpha)
            + 0.5 * Dii * np.dot(self.alpha, self.alpha)
        )
        elapsed = time.time() - start
        if not self.fobj_history or self.fobj_history[-1][1] != total_step:
            self.fobj_history.append((elapsed, total_step, final_obj))
        if not self.fobj_history_cd or self.fobj_history_cd[-1][1] != total_cd_step:
            self.fobj_history_cd.append((elapsed, total_cd_step, final_obj))

        self.w = w

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.sign(X @ self.w)
    
    
if __name__ == "__main__":
    import os
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    print("Modello utilizzato: SVM con two-variable Coordinate Descent Duale")

    # Percorsi dei file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "dataset", "a9a.txt")
    test_path = os.path.join(base_dir, "dataset", "a9a_t.txt")

    # Caricamento dati
    X_train, y_train, X_test, y_test, X_all, y_all = load_data(train_path, test_path)
    print(f"Train set:    {X_train.shape[0]} campioni, {X_train.shape[1]} feature")
    print(f"Test set:     {X_test.shape[0]} campioni, {X_test.shape[1]} feature")
    print(f"Dataset totale: {X_all.shape[0]} campioni")

    # # Addestramento
    # svm_duale = SVM_2CD(C=8192, n_iters=1000, tol = 1e-4)
    # svm_duale.fit(X_train, y_train)

    # # Predizione e accuratezza
    # y_pred = svm_duale.predict(X_test)
    # accuracy = np.mean(y_pred == y_test)
    # print(f"Accuracy sul test set: {accuracy * 100:.2f}%")

    # sv = (svm_duale.alpha > 1e-5) & (svm_duale.alpha < svm_duale.C - 1e-5)
    # print(f"Support vectors: {sv.sum()} su {X_train.shape[0]} campioni")

    # # accuracy su train set
    # y_pred_train = svm_duale.predict(X_train)
    # print(f"Accuracy train: {np.mean(y_pred_train == y_train) * 100:.2f}%")
    print("Addestramento SVM sklearn...")
    svm_sk = LinearSVC(C=8192, max_iter=1000, tol=1e-1, dual=True, loss="squared_hinge", fit_intercept=True)
    svm_sk.fit(X_train, y_train)

    y_pred_sk = svm_sk.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    print(f"sklearn Accuracy: {acc_sk * 100:.2f}%")