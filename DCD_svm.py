import numpy as np
import matplotlib.pyplot as plt
from utils import load_data
from tqdm import tqdm
import time

class SVM_DCD:
    def __init__(self, C=1.0, n_iters=1000, tol = 1e-6):
        self.C = C
        self.n_iters = n_iters
        self.tol = tol

        self.alpha = None
        self.w = None

        self.fobj_history = []

    def fit(self, X, y):
        X = np.hstack([X, np.ones((X.shape[0], 1))]) #Aggiungiamo una colonna di 1 per il bias

        n_samples, n_features = X.shape

        y = np.where(y <= 0, -1, 1).astype(float)

        self.alpha = np.zeros(n_samples)
        w = np.zeros(n_features)
        self.fobj_history = []

        Q_diag = np.sum(X**2, axis=1) #Matrice diagonale denominatore per l'aggiornamento

        start = time.time()

        for epoch in tqdm(range(self.n_iters), desc="Epoche", unit="epoch"):
            n_updates = 0

            for i in np.random.permutation(n_samples):
                #Calcolo del gradiente
                G = y[i] * np.dot(w, X[i]) - 1

                #Aggiorniamo la variabile duale alpha_i

                if self.alpha[i] == 0 and G >= 0:
                    continue
                if self.alpha[i] == self.C and G <= 0:
                    continue

                alpha_new = self.alpha[i] - G / Q_diag[i]
                alpha_new = float(np.clip(alpha_new, 0, self.C))

                delta = alpha_new - self.alpha[i] #Aggiornamento della variabile duale di un fattore d 
                if abs(delta) > 1e-12: #Se l'aggiornamento è significativo, aggiorniamo i pesi w
                    w += delta * y[i] * X[i] #Aggiornamento incrementale dei pesi w
                    n_updates += 1
                self.alpha[i] = alpha_new
            
            #Calcoliamo il valore della funzione obiettivo f(α) = (1/2)||w||² - Σα_i
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
    print("Modello utilizzato: SVM con Coordinate Descent Duale")

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
    svm_duale = SVM_DCD(C=8.192, n_iters=1000, tol = 1e-4)
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
