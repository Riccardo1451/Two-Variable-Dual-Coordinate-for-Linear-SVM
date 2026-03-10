import numpy as np
import matplotlib.pyplot as plt
from utils import load_data

class SVM:
    def __init__(self, C= 1.0 ,learning_rate=0.01, n_iters=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y): #Assumendo che X sia un array in numpy 
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1) #Conversione delle etichette in -1 e 1
        self.w = np.zeros(n_features) #Pesi inizializzati 
        self.b = 0 #Fattore di Bias

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X): #Iterazione su ogni campione 
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1 #Condizione di margine y * (w * x + b) >= 1
                #Se la condizione è vera applichiamo le regole di aggiornamento dei pesi e del bias
                if condition:
                    self.w -= self.learning_rate  * self.w
                    #Il Bias non si aggiorna poichè in questo caso il gradiente è zero
                else:
                    self.w -= self.learning_rate * (self.w - self.C * np.dot(x_i, y_[idx]))
                    self.b += self.learning_rate * self.C * y_[idx]
        #Alla fine di questo metodo avremo i pesi e il bias addestrati

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b #Questo renderà o 1 o -1 a seconda del segno del risultato
        return np.sign(approx)


if __name__ == "__main__":
    import os
    print("Modello utilizzato: SVM")

    # Percorsi dei file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "dataset", "a1a.txt")
    test_path = os.path.join(base_dir, "dataset", "a1a_t.txt")

    # Caricamento dati
    X_train, y_train, X_test, y_test, X_all, y_all = load_data(train_path, test_path)
    print(f"Train set:    {X_train.shape[0]} campioni, {X_train.shape[1]} feature")
    print(f"Test set:     {X_test.shape[0]} campioni, {X_test.shape[1]} feature")
    print(f"Dataset totale: {X_all.shape[0]} campioni")

    # Addestramento
    svm = SVM(C=20.0, learning_rate=0.001, n_iters=1000)
    svm.fit(X_train, y_train)

    # Predizione e accuratezza
    y_pred = svm.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy sul test set: {accuracy * 100:.2f}%")
