import numpy as np
from tqdm import tqdm
from utils import load_data

'''Per evitare confusioni, questo metodo muove due variabili duali alla volta, questo perchè il termine di bias b rimane libero nel primale
e quindi, una volta passati alla formulazione duale, troviamo il vincolo sulla somma delle variabili duali, se vogliamo aggiornare alpha_i,
per rispettare il vincolo, dobbiamo aggiornare anche un'altra variabile duale alpha_j in modo che la somma rimanga zero.
In sintesi stiamo svolgendo l'algoritmo SMO.'''

class SVM_Duale:
    def __init__(self, C=1.0, n_iters=1000, tol = 1e-6):
        self.C = C
        self.n_iters = n_iters
        self.tol = tol

        self.alpha = None #Variabili duali, sono i coefficienti Lagrangiani
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        y = np.where(y <= 0, -1, 1)

        self.alpha = np.zeros(n_samples) #Inizializziamo le variabili duali a zero
        self.b = 0

        K = np.dot(X, X.T) #Matrice dei kernel lineare, shape (n_samples, n_samples)

        # Cache degli errori: E_cache[i] = f(x_i) - y_i
        E_cache = -y.copy()

        for epoch in tqdm(range(self.n_iters), desc="Epoche", unit="epoch"):
            n_updates = 0

            idxs = np.random.permutation(n_samples) #Shuffle per il training

            for i in idxs :
                # leggiamo E_i dalla cache invece di ricalcolarlo 
                E_i = E_cache[i]

                #Controlliamo le condizione di KKT per decidere se aggiornare o meno la variabile duale
                kkt_violated = (
                    (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or
                    (y[i] * E_i > self.tol and self.alpha[i] > 0)
                )

                if not kkt_violated:
                    continue
                    
                #Scelta di un secondo indice j diverso da i, scegliamo il j che massimizza la violazione KKT
                # leggiamo E_all dalla cache
                violators = np.where(
                    ((y * E_cache < -self.tol) & (self.alpha < self.C)) |
                    ((y * E_cache >  self.tol) & (self.alpha > 0)))[0]
                violators = violators[violators != i]

                if len(violators) > 0:
                    j = violators[np.argmax(np.abs(E_i - E_cache[violators]))]
                else:
                    j = (i + 1) % n_samples
                    while j == i:
                        j = (j + 1) % n_samples
                
                E_j = E_cache[j]

                alpha_i_old = self.alpha[i]
                alpha_j_old = self.alpha[j]

                #Applichiamo il bound [L,H] per le variabili duali
                if y[i] == y[j]:
                    L = max(0.0, self.alpha[i] + self.alpha[j] - self.C)
                    H = min(self.C, self.alpha[i] + self.alpha[j])
                else:
                    L = max(0.0, self.alpha[i] - self.alpha[j])
                    H = min(self.C, self.C + self.alpha[i] - self.alpha[j])
                if L >= H:
                    continue

                eta = K[i,i]+K[j,j]-2*K[i,j]
                if eta <= 0:
                    continue

                #Aggiornamento di alpha_i in forma chiusa
                self.alpha[i] = alpha_i_old + y[i] * (E_j - E_i) / eta
                self.alpha[i] = float(np.clip(self.alpha[i], L, H))

                #Aggiormento forzato dal vincolo di alpha_j
                self.alpha[j] = alpha_j_old + y[i] * y[j] * (alpha_i_old - self.alpha[i])

                #Aggiornamento del bias b
                b1 = (self.b - E_i 
                    - y[i] * (self.alpha[i] - alpha_i_old) * K[i,i] 
                    - y[j] * (self.alpha[j] - alpha_j_old) * K[i,j])
                b2 = (self.b - E_j 
                      - y[i] * (self.alpha[i] - alpha_i_old) * K[i,j] 
                      - y[j] * (self.alpha[j] - alpha_j_old) * K[j,j])
                
                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2

                # aggiorniamo la cache solo per i e j
                E_cache[i] = (self.alpha * y) @ K[:, i] + self.b - y[i]
                E_cache[j] = (self.alpha * y) @ K[:, j] + self.b - y[j]
                
                n_updates += 1

            if n_updates == 0:
                print(f"Epoch {epoch}: nessun aggiornamento, convergenza raggiunta con {n_updates} aggiornamenti.")
                break
            
        #Calcoliamo il peso w a partire dalle variabili duali
        self.w = (self.alpha * y) @ X

        #Calcoliamo b dal numero di support vectors
        sv = np.where((self.alpha > 1e-5) & (self.alpha < self.C - 1e-5))
        if np.any(sv):
            self.b = np.mean(y[sv] - X[sv] @ self.w)
        else:
            self.b = np.mean(y - X @ self.w)

    def predict(self, X):
        approx = X @ self.w + self.b #Il bias è positivo perchè nella formula del duale è con segno opposto
        return np.sign(approx)

if __name__ == "__main__":
    import os
    

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
    svm_duale = SVM_Duale(C=1.0, n_iters=1000, tol = 1e-1)
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


    # from sklearn.svm import LinearSVC
    # from sklearn.metrics import accuracy_score

    # svm_sk = LinearSVC(C=1.0, max_iter=10000)
    # svm_sk.fit(X_train, y_train)
    # y_pred_sk = svm_sk.predict(X_test)
    # print(f"sklearn accuracy: {accuracy_score(y_test, y_pred_sk) * 100:.2f}%")