import numpy as np 
from utils import load_data
from tqdm import tqdm
import time

class SVM_2CD:
    def __init__(self, C=1.0, n_iters=1000, tol = 1e-6):
        self.C = C
        self.n_iters = n_iters
        self.tol = tol

        self.alpha = None
        self.w = None
        self.fobj_history = []
    
    def _solve_2d_subproblem(self, ai, aj, Gi, Gj, Qii, Qjj, Qij):
        best_val = np.inf
        best_di, best_dj = 0.0, 0.0
        C = self.C

        def obj(di, dj):
            return 0.5*(Qii*di**2 + 2*Qij*di*dj + Qjj*dj**2) + Gi*di + Gj*dj

        # caso 1: minimo interno
        det = Qii*Qjj - Qij**2
        if abs(det) > 1e-12:
            di = (-Gi*Qjj + Gj*Qij) / det
            dj = (-Gj*Qii + Gi*Qij) / det
            if 0 <= ai+di <= C and 0 <= aj+dj <= C:
                val = obj(di, dj)
                if val < best_val:
                    best_val, best_di, best_dj = val, di, dj

        # caso 2: αᵢ = 0
        di = -ai
        dj = np.clip(-(Gj + Qij*di) / Qjj, -aj, C-aj)
        val = obj(di, dj)
        if val < best_val:
            best_val, best_di, best_dj = val, di, dj

        # caso 3: αᵢ = C
        di = C - ai
        dj = np.clip(-(Gj + Qij*di) / Qjj, -aj, C-aj)
        val = obj(di, dj)
        if val < best_val:
            best_val, best_di, best_dj = val, di, dj

        # caso 4: αⱼ = 0
        dj = -aj
        di = np.clip(-(Gi + Qij*dj) / Qii, -ai, C-ai)
        val = obj(di, dj)
        if val < best_val:
            best_val, best_di, best_dj = val, di, dj

        # caso 5: αⱼ = C
        dj = C - aj
        di = np.clip(-(Gi + Qij*dj) / Qii, -ai, C-ai)
        val = obj(di, dj)
        if val < best_val:
            best_val, best_di, best_dj = val, di, dj

        return best_di, best_dj
    
    def fit(self, X, y):
        X = np.hstack([X, np.ones((X.shape[0], 1))]) #Aggiungiamo una colonna di 1 per il bias

        y = np.where(y <= 0, -1, 1).astype(float)

        n_samples, n_features = X.shape

        self.alpha = np.zeros(n_samples)
        w = np.zeros(n_features)
        self.fobj_history = []

        Q_diag = np.sum(X**2, axis=1)

        start = time.time()

        for epoch in tqdm(range(self.n_iters), desc="Epoche", unit="epoch"):
            n_updates = 0
            perm = np.random.permutation(n_samples)

            for k in range(0, n_samples-1, 2):
                i,j = perm[k], perm[k+1] #Permutazione a coppie consecutive

                #Calcolo del gradiente
                Gi = y[i] * np.dot(w, X[i]) - 1
                Gj = y[j] * np.dot(w, X[j]) - 1

                Qii = Q_diag[i]
                Qjj = Q_diag[j]
                Qij = y[i] * y[j] * np.dot(X[i], X[j])

                di,dj = self._solve_2d_subproblem(
                    self.alpha[i], self.alpha[j], 
                    Gi, Gj, Qii, Qjj, Qij
                )
                
                if abs(di) > 1e-12 or abs(dj) > 1e-12:
                    w += di * y[i] * X[i] + dj * y[j] * X[j]
                    n_updates += 1
                
                self.alpha[i] += di
                self.alpha[j] += dj
                    
            fobj = 0.5 * np.dot(w,w) - self.alpha.sum()
            self.fobj_history.append((time.time() - start,fobj))
            if n_updates == 0 :
                print(f"Convergenza raggiunta all'epoca {epoch}")
                break

        self.w = w

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.sign(X @ self.w)
    
if __name__ == "__main__":
    import os
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

    # Addestramento
    svm_duale = SVM_2CD(C=8.192, n_iters=1000, tol = 1e-4)
    svm_duale.fit(X_train, y_train)

    # Predizione e accuratezza
    y_pred = svm_duale.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy sul test set: {accuracy * 100:.2f}%")

    sv = (svm_duale.alpha > 1e-5) & (svm_duale.alpha < svm_duale.C - 1e-5)
    print(f"Support vectors: {sv.sum()} su {X_train.shape[0]} campioni")

    # accuracy su train set
    y_pred_train = svm_duale.predict(X_train)
    print(f"Accuracy train: {np.mean(y_pred_train == y_train) * 100:.2f}%")