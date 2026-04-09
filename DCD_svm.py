import numpy as np
import matplotlib.pyplot as plt
from utils import load_data
from tqdm import tqdm
import time

import numpy as np
import time
from tqdm import tqdm

class SVM_DCD:
    def __init__(self, C=1.0, n_iters=1000, tol=1e-6):
        self.C = C
        self.n_iters = n_iters
        self.tol = tol

        self.alpha = None
        self.w = None
        self.fobj_history = []
        self.fobj_history_cd = []

    def fit(self, X, y):
        # Bias embedding (come nel paper: no vincolo duale)
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        y = np.where(y <= 0, -1, 1).astype(float)
        n_samples, n_features = X.shape

        self.alpha = np.zeros(n_samples)
        w = np.zeros(n_features)

        # L2-SVM: termine diagonale
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

            for i in perm:
                total_cd_step += 1

                # Gradiente
                G = y[i] * np.dot(w, X[i]) - 1 + Dii * self.alpha[i]

                # Gradiente proiettato (bound solo inferiore: 0)
                if self.alpha[i] == 0:
                    PG = min(0.0, G)
                else:
                    PG = G

                M = max(M, PG)
                m = min(m, PG)

                # Skip se ottimo
                if self.alpha[i] == 0 and G >= 0:
                    continue

                # Update closed-form
                alpha_old = self.alpha[i]
                alpha_new = alpha_old - G / Q_diag[i]
                alpha_new = max(0.0, alpha_new)  # bound inferiore

                delta = alpha_new - alpha_old

                # Update incrementale di w
                if abs(delta) > 1e-12:
                    w += delta * y[i] * X[i]
                    self.alpha[i] = alpha_new

                total_step += 1

                # Logging funzione obiettivo
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

            # Criterio di stop teorico (paper)
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
        scores = X @ self.w
        return np.where(scores >= 0, 1, -1)
