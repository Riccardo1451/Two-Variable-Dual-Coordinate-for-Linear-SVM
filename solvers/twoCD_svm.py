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
        """
        Solves the 2D subproblem for the pair of coordinates (i, j) in Two-CD.
        For L2-SVM: upper bound = inf, so box = [0, inf) x [0, inf)
        """
        delta = Qii * Qjj - Qij ** 2

        # Case degenerate: when delta is zero and H is not positive definite, but should't happen in L2-SVM
        if delta <= 1e-10:
            di, dj = 0.0, 0.0
            if not (ai == 0 and Gi >= 0):
                di = max(0.0, ai - Gi / Qii) - ai
            if not (aj == 0 and Gj >= 0):
                dj = max(0.0, aj - Gj / Qjj) - aj
            return di, dj

        # Step 1: free solution without box contraints
        ai_free = ai + (-Qjj * Gi + Qij * Gj) / delta
        aj_free = aj + (-Qii * Gj + Qij * Gi) / delta

        # Step 2: if free solution is feasible we can stop
        if ai_free >= 0 and aj_free >= 0:
            return ai_free - ai, aj_free - aj

        # Step 3: otherwise, projecting free solution onto the box
        ai_proj = max(0.0, ai_free)
        aj_proj = max(0.0, aj_free)

        # Step 4: check opt condition on ai_proj (Th II.1 + II.2)
        # if ai_proj is on the lower bound (= 0), check gradient sign
        use_j = True
        if ai_proj <= 0:
            grad_check = Qii * (ai_proj - ai) + Qij * (aj_proj - aj) + Gi
            if grad_check >= 0:
                # opt reached so we can fix ai_proj and optimize aj in 1D
                use_j = False

        if not use_j:
            # fix and optimize aj in 1D
            ai_bar = ai_proj
            aj_bar = max(0.0, aj - (Qij * (ai_bar - ai) + Gj) / Qjj)
        else:
            # For Th II.3: opt on aj reached
            # fix aj_bar = aj_proj optimize ai in 1D
            aj_bar = aj_proj
            ai_bar = max(0.0, ai - (Qij * (aj_bar - aj) + Gi) / Qii)

        return ai_bar - ai, aj_bar - aj

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

        for epoch in tqdm(range(self.n_iters), desc="Epochs", unit="epoch"):
            M = -np.inf
            m = np.inf

            perm = np.random.permutation(n_samples)

            # Following permutation scheme 2.13
            for k in range(0, n_samples - 1, 2):
                i, j = perm[k], perm[k + 1]
                total_cd_step += 2

                # Compute Gradients
                Gi = y[i] * np.dot(w, X[i]) - 1 + Dii * self.alpha[i]
                Gj = y[j] * np.dot(w, X[j]) - 1 + Dii * self.alpha[j]

                # Projected Gradients
                PGi = min(0.0, Gi) if self.alpha[i] == 0 else Gi
                PGj = min(0.0, Gj) if self.alpha[j] == 0 else Gj

                M = max(M, PGi, PGj)
                m = min(m, PGi, PGj)

                # skipping if both proj are zero (opt condition reached)
                if (self.alpha[i] == 0 and Gi >= 0) and (self.alpha[j] == 0 and Gj >= 0):
                    continue

                Qii = Q_diag[i]
                Qjj = Q_diag[j]
                Qij = y[i] * y[j] * np.dot(X[i], X[j])

                # Call the 2D subproblem solver following algorithm II from appendix
                di, dj = self._solve_2d_subproblem(
                    self.alpha[i], self.alpha[j],
                    Gi, Gj, Qii, Qjj, Qij
                )

                if abs(di) > 1e-12 or abs(dj) > 1e-12:
                    w += di * y[i] * X[i] + dj * y[j] * X[j]
                    self.alpha[i] += di
                    self.alpha[j] += dj

                total_step += 1

                # Logging of relative obj gap 
                if total_step % LOG_INTERVAL == 0:
                    fobj = (
                        0.5 * np.dot(w, w)
                        - np.sum(self.alpha)
                        + 0.5 * Dii * np.dot(self.alpha, self.alpha)
                    )
                    self.fobj_history.append(
                        (time.time() - start, total_step, fobj)
                    )

                # Logging of CD steps tried
                if total_cd_step % LOG_INTERVAL == 0:
                    fobj = (
                        0.5 * np.dot(w, w)
                        - np.sum(self.alpha)
                        + 0.5 * Dii * np.dot(self.alpha, self.alpha)
                    )
                    self.fobj_history_cd.append(
                        (time.time() - start, total_cd_step, fobj)
                    )
                    

            # Stopping Criterion 
            if M - m < self.tol:
                print(f"\nConvergence reached at epoch {epoch}")
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