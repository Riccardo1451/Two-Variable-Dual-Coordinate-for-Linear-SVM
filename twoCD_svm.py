import numpy as np 
from utils import load_data
from tqdm import tqdm
import time

class SVM_2CD:
    def __init__(self, C=1.0, n_iters=1000, tol=1e-6, solve_method="constrained"):
        """
        solve_method: 
            - "naive": soluzione libera con proiezione semplice (può divergere)
            - "constrained": soluzione vincolata robusta (consigliato)
        """
        self.C = C
        self.n_iters = n_iters
        self.tol = tol
        self.solve_method = solve_method

        self.alpha = None
        self.w = None
        self.fobj_history = []
        self.fobj_history_cd = []

    def _solve_2d_subproblem_naive(self, ai, aj, Gi, Gj, Qii, Qjj, Qij):
        """
        Soluzione NAIVE: sistema libero con proiezione semplice.
        PROBLEMATICO: proiezione scorretta quando la soluzione esce dai vincoli.
        """
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

        # proiezione semplice (scorretta se viola i vincoli in modo non simmetrico)
        ai_new = max(0.0, ai_new)
        aj_new = max(0.0, aj_new)

        return ai_new - ai, aj_new - aj

    def _solve_2d_subproblem_constrained(self, ai, aj, Gi, Gj, Qii, Qjj, Qij):
        """
        Solve the 2D quadratic subproblem in delta-space with lower bounds:
            di >= -ai, dj >= -aj.
        We evaluate feasible KKT candidates (free point, two edges, corner)
        and pick the one with the lowest local objective.
        """

        def local_obj(di, dj):
            return (
                Gi * di
                + Gj * dj
                + 0.5 * (Qii * di * di + 2.0 * Qij * di * dj + Qjj * dj * dj)
            )

        # Start from "no move" candidate to avoid accidental ascent.
        best_di, best_dj = 0.0, 0.0
        best_obj = 0.0

        delta = Qii * Qjj - Qij ** 2

        # Candidate 1: unconstrained minimizer (if numerically stable and feasible)
        if delta > 1e-12:
            di_free = (-Qjj * Gi + Qij * Gj) / delta
            dj_free = (-Qii * Gj + Qij * Gi) / delta
            if di_free >= -ai and dj_free >= -aj:
                obj_free = local_obj(di_free, dj_free)
                if obj_free < best_obj:
                    best_di, best_dj = di_free, dj_free
                    best_obj = obj_free

        # Candidate 2: edge di = -ai, optimize dj with bound dj >= -aj
        di_edge = -ai
        dj_star = -(Gj + Qij * di_edge) / Qjj
        dj_edge = max(dj_star, -aj)
        obj_edge_i = local_obj(di_edge, dj_edge)
        if obj_edge_i < best_obj:
            best_di, best_dj = di_edge, dj_edge
            best_obj = obj_edge_i

        # Candidate 3: edge dj = -aj, optimize di with bound di >= -ai
        dj_edge = -aj
        di_star = -(Gi + Qij * dj_edge) / Qii
        di_edge = max(di_star, -ai)
        obj_edge_j = local_obj(di_edge, dj_edge)
        if obj_edge_j < best_obj:
            best_di, best_dj = di_edge, dj_edge
            best_obj = obj_edge_j

        # Candidate 4: corner di = -ai, dj = -aj
        obj_corner = local_obj(-ai, -aj)
        if obj_corner < best_obj:
            best_di, best_dj = -ai, -aj

        return best_di, best_dj

    def _solve_2d_subproblem(self, ai, aj, Gi, Gj, Qii, Qjj, Qij):
        """Wrapper che seleziona il metodo di risoluzione."""
        if self.solve_method == "constrained":
            return self._solve_2d_subproblem_constrained(ai, aj, Gi, Gj, Qii, Qjj, Qij)
        else:  # "naive" o altro
            return self._solve_2d_subproblem_naive(ai, aj, Gi, Gj, Qii, Qjj, Qij)

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
        scores = X @ self.w
        return np.where(scores >= 0, 1, -1)