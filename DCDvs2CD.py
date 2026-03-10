import os 
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data
from DCD_svm import SVM_DCD
from twoCD_svm import SVM_2CD

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

base_dir = os.path.dirname(os.path.abspath(__file__))
datasets = {
    "a9a": ("a9a.txt", "a9a_t.txt"),
    "a1a": ("a1a.txt", "a1a_t.txt"),
    "a6a": ("a6a.txt", "a6a_t.txt")
}

C_values = [0.1, 1.0, 8.192]
TOL = 1e-4

for dataset_name, (train_file, test_file) in datasets.items():
    train_path = os.path.join(base_dir, "dataset", train_file)
    test_path = os.path.join(base_dir, "dataset", test_file)

    X_train, y_train, X_test, y_test, _, _ = load_data(train_path, test_path)

    print(f"\nDataset: {dataset_name}- Train: {X_train.shape[0]} campioni, {X_train.shape[1]} feature - Test: {X_test.shape[0]} campioni, {X_test.shape[1]} feature")

    for C in C_values:
        print(f"\n C = {C}")

        #----------DCD---------
        dcd = SVM_DCD(C=C, n_iters=1000, tol=TOL)
        dcd.fit(X_train, y_train)
        y_pred_dcd = dcd.predict(X_test)
        acc_dcd = accuracy_score(y_test, y_pred_dcd)
        print(f"SVM DCD Accuracy: {acc_dcd * 100:.2f}%")

        #----------Two-CD---------
        two_cd = SVM_2CD(C=C, n_iters=1000, tol=TOL)
        two_cd.fit(X_train, y_train)
        y_pred_2cd = two_cd.predict(X_test)
        acc_2cd = accuracy_score(y_test, y_pred_2cd)
        print(f"SVM Two-CD Accuracy: {acc_2cd * 100:.2f}%")

        #----------sklearn---------
        svm_sk = LinearSVC(C=C, max_iter=10000, tol=TOL, dual="auto")
        svm_sk.fit(X_train, y_train)
        y_pred_sk = svm_sk.predict(X_test)
        acc_sk = accuracy_score(y_test, y_pred_sk)
        print(f"sklearn Accuracy: {acc_sk * 100:.2f}%")

        #----------plot obj vs time---------
        times_dcd,   obj_dcd   = zip(*dcd.fobj_history)
        times_2cd,   obj_2cd   = zip(*two_cd.fobj_history)

        f_star = min(obj_dcd[-1], obj_2cd[-1]) #Normalizzazione rispetto al valore finale più basso tra i due metodi

        rel_dcd  = [(f - f_star) / abs(f_star) for f in obj_dcd]
        rel_2cd  = [(f - f_star) / abs(f_star) for f in obj_2cd]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(times_dcd,  rel_dcd,  label="DCD (1-var)",  linestyle="--")
        ax.semilogy(times_2cd,  rel_2cd,  label="2-CD (2-var)", linestyle="-")
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Relative objective gap  |f(α) - f*| / |f*|")
        ax.set_title(f"{dataset_name}  —  C={C}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(base_dir, "results", f"plot_{dataset_name}_C{C}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Grafico salvato: {plot_path}")