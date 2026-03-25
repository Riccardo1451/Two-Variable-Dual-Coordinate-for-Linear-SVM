import os 
import json
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
}

C_values = [8192, 1]
TOL = 1e-1
SEED = 42

# Impostazioni per confronto piu' vicino al paper.
USE_STANDARD_SCALING = False
REFERENCE_ITERS = 4000
REFERENCE_TOL = 1e-2

# Scelta asse step:
# - "cd": usa il numero di passi CD tentati (consigliato per confronto con il paper)
# - "effective": usa solo gli update effettivi (comportamento precedente)
STEP_AXIS_MODE = "cd"

# Cache persistente di f* per evitare ricalcolo ad ogni run.
FSTAR_CACHE_PATH = os.path.join(base_dir, "results", "fstar_cache.json")

# Override manuale opzionale: se presente, viene usato questo valore.
# Formato chiave: (dataset_name, C)
FSTAR_OVERRIDE = {
    #("a9a", 8192): -13742.373305,
}

def compute_relative_gap(obj_values, f_star):
    denom = max(abs(f_star), 1e-16)
    return [max((f - f_star) / denom, 1e-16) for f in obj_values]


def select_history(model):
    if STEP_AXIS_MODE == "cd":
        return model.fobj_history_cd
    return model.fobj_history


def load_fstar_cache(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: float(v) for k, v in raw.items()}
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def save_fstar_cache(path, cache):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def fstar_key(dataset_name, C):
    return f"{dataset_name}|C={C}"


def compute_reference_fstar(X_train, y_train, C):
    np.random.seed(SEED)
    ref_dcd = SVM_DCD(C=C, n_iters=REFERENCE_ITERS, tol=REFERENCE_TOL)
    ref_dcd.fit(X_train, y_train)

    np.random.seed(SEED)
    ref_2cd = SVM_2CD(C=C, n_iters=REFERENCE_ITERS, tol=REFERENCE_TOL)
    ref_2cd.fit(X_train, y_train)

    best_dcd = min(v for _, _, v in ref_dcd.fobj_history)
    best_2cd = min(v for _, _, v in ref_2cd.fobj_history)
    return min(best_dcd, best_2cd)


fstar_cache = load_fstar_cache(FSTAR_CACHE_PATH)

for dataset_name, (train_file, test_file) in datasets.items():
    train_path = os.path.join(base_dir, "dataset", train_file)
    test_path = os.path.join(base_dir, "dataset", test_file)

    X_train, y_train, X_test, y_test, _, _ = load_data(
        train_path,
        test_path,
        use_scaling=USE_STANDARD_SCALING,
    )

    print(f"\nDataset: {dataset_name}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    for C in C_values:
        print(f"\nC = {C}")

        # =========================
        # CALCOLO f* 
        # =========================
        key = fstar_key(dataset_name, C)
        override_key = (dataset_name, C)
        if override_key in FSTAR_OVERRIDE:
            f_star = float(FSTAR_OVERRIDE[override_key])
            print("Uso f* da override manuale.")
            fstar_cache[key] = f_star
        elif key in fstar_cache:
            f_star = float(fstar_cache[key])
            print("Uso f* da cache.")
        else:
            print("Calcolo f* di riferimento (robusto)...")
            f_star = compute_reference_fstar(X_train, y_train, C)
            fstar_cache[key] = f_star
            save_fstar_cache(FSTAR_CACHE_PATH, fstar_cache)

        print(f"f* ≈ {f_star:.6f}")

        # =========================
        # DCD
        # =========================
        print("Addestramento SVM DCD...")
        np.random.seed(SEED)
        dcd = SVM_DCD(C=C, n_iters=3000, tol=TOL)
        dcd.fit(X_train, y_train)

        y_pred_dcd = dcd.predict(X_test)
        acc_dcd = accuracy_score(y_test, y_pred_dcd)
        print(f"DCD Accuracy: {acc_dcd * 100:.2f}%")

        # =========================
        # 2-CD
        # =========================
        print("Addestramento SVM Two-CD...")
        np.random.seed(SEED)
        two_cd = SVM_2CD(C=C, n_iters=2000, tol=TOL, solve_method="constrained")
        two_cd.fit(X_train, y_train)

        y_pred_2cd = two_cd.predict(X_test)
        acc_2cd = accuracy_score(y_test, y_pred_2cd)
        print(f"2-CD Accuracy: {acc_2cd * 100:.2f}%")

        # =========================
        # sklearn (solo controllo)
        # =========================
        print("Addestramento SVM sklearn...")
        svm_sk = LinearSVC(C=C, max_iter=5000, tol=TOL, dual=True, loss="squared_hinge", fit_intercept=False)
        svm_sk.fit(X_train, y_train)

        y_pred_sk = svm_sk.predict(X_test)
        acc_sk = accuracy_score(y_test, y_pred_sk)
        print(f"sklearn Accuracy: {acc_sk * 100:.2f}%")

        # =========================
        # ESTRAZIONE DATI
        # =========================
        hist_dcd = select_history(dcd)
        hist_2cd = select_history(two_cd)
        times_dcd, steps_dcd, obj_dcd = zip(*hist_dcd)
        times_2cd, steps_2cd, obj_2cd = zip(*hist_2cd)

        if STEP_AXIS_MODE == "effective":
            # In modalita' effective, un passo 2-CD aggiorna 2 variabili.
            steps_2cd = [s * 2 for s in steps_2cd]

        # =========================
        # RELATIVE GAP
        # =========================
        rel_dcd = compute_relative_gap(obj_dcd, f_star)
        rel_2cd = compute_relative_gap(obj_2cd, f_star)

        # =========================
        # PLOT
        # =========================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Dataset: {dataset_name}   |   C = {C}",
            fontsize=13,
            fontweight="bold",
        )

        # ---- vs step ----
        ax1.semilogy(steps_dcd, rel_dcd, label="OneVariableDCD", linestyle="--")
        ax1.semilogy(steps_2cd, rel_2cd, label="TwoVariable-DCD", linestyle="-")

        if STEP_AXIS_MODE == "cd":
            ax1.set_xlabel("numero di passi CD tentati")
        else:
            ax1.set_xlabel("numero di aggiornamenti effettivi (normalizzati)")
        ax1.set_ylabel("funzione obiettivo relativa")
        ax1.set_title("Convergenza rispetto ai passi")
        ax1.legend(title="Metodo")
        ax1.grid(True, which="both", alpha=0.3)

        # ---- vs tempo ----
        ax2.semilogy(times_dcd, rel_dcd, label="OneVariableDCD", linestyle="--")
        ax2.semilogy(times_2cd, rel_2cd, label="TwoVariable-DCD", linestyle="-")

        ax2.set_xlabel("tempo di esecuzione (s)")
        ax2.set_ylabel("funzione obiettivo relativa")
        ax2.set_title("Convergenza rispetto al tempo")
        ax2.legend(title="Metodo")
        ax2.grid(True, which="both", alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        plot_path = os.path.join(base_dir, "results", f"plot_{dataset_name}_C{C}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Grafico salvato in: {plot_path}")