import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score
from joblib import Parallel, delayed

from utils.data_loader import load_and_preprocess_dns_data

OUT_DIR = "heavy_master_results"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = f"{OUT_DIR}/progress.log"

def log_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

N_POP = 100
N_GEN = 100

def get_fitness(ind, X_tr, y_tr, X_va, y_va):
    selected = [i for i, bit in enumerate(ind) if bit == 1]
    if len(selected) == 0: return 0.0
    # Light proxy RF for wrapper evaluation speed during 10,000 queries
    clf = RandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=1, random_state=42)
    clf.fit(X_tr[:, selected], y_tr)
    return f1_score(y_va, clf.predict(X_va[:, selected]), average='weighted')

def run_tournament(X_train, y_train, X_val, y_val):
    n_features = X_train.shape[1]
    algorithms_to_run = ["Proposed-Penalty", "Matrix-GA", "JAYA-GA"]
    best_features_dict = {}

    # Downsample for GA search to make runtime feasible locally (5% of train) 
    # Wrapper FS uses small subsets to evaluate quickly
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_train), int(len(X_train) * 0.05), replace=False)
    X_tr_sub, y_tr_sub = X_train[sample_idx], y_train[sample_idx]

    def eval_pop(pop, algo_name):
        def _eval(ind):
            f1 = get_fitness(ind, X_tr_sub, y_tr_sub, X_val, y_val)
            if algo_name == "Proposed-Penalty":
                penalty = sum(ind) / n_features
                return f1 - (0.01 * penalty)
            return f1
        return Parallel(n_jobs=-1)(delayed(_eval)(ind) for ind in pop)

    for algo in algorithms_to_run:
        log_print(f"\n🚀 Running Heavy {algo} Engine (Pop={N_POP}, Gen={N_GEN})...")
        pop = np.random.randint(0, 2, (N_POP, n_features))
        best_ind = None
        global_best_fit = -float('inf')

        start_time = time.time()
        for gen in range(N_GEN):
            fits = eval_pop(pop, algo)
            
            gen_best_idx = np.argmax(fits)
            if fits[gen_best_idx] > global_best_fit:
                global_best_fit = fits[gen_best_idx]
                best_ind = pop[gen_best_idx].copy()

            if algo == "JAYA-GA":
                worst_ind = pop[np.argmin(fits)]
                r1, r2 = np.random.rand(N_POP, n_features), np.random.rand(N_POP, n_features)
                new_pop = pop + r1*(best_ind - pop) - r2*(worst_ind - pop)
                pop = np.clip(np.round(new_pop), 0, 1).astype(int)
            elif algo == "Matrix-GA":
                np.random.shuffle(pop)
                for i in range(0, N_POP-1, 2):
                    mask = np.random.rand(n_features) > 0.5
                    pop[i][mask], pop[i+1][mask] = pop[i+1][mask], pop[i][mask]
                mut_mask = np.random.rand(N_POP, n_features) < 0.05
                pop[mut_mask] ^= 1
            else:
                new_pop = []
                for _ in range(N_POP // 2):
                    p1, p2 = pop[np.random.randint(0, N_POP)], pop[np.random.randint(0, N_POP)]
                    child = np.where(np.random.rand(n_features) > 0.5, p1, p2)
                    if np.random.rand() < 0.1: child[np.random.randint(0, n_features)] ^= 1
                    new_pop.extend([child, best_ind]) 
                pop = np.array(new_pop[:N_POP])
                
            if (gen+1) % 10 == 0:
                elapsed = time.time() - start_time
                log_print(f"   [{algo}] Generation {gen+1}/{N_GEN}... Best Fitness: {global_best_fit:.4f} (Elapsed: {elapsed:.1f}s)")

        best_features_dict[algo] = [i for i, b in enumerate(best_ind) if b == 1]
        log_print(f"   -> Found subset with {len(best_features_dict[algo])} features. Fitness: {global_best_fit:.4f}")

    return best_features_dict

def main():
    log_print("Loading Real Data...")
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names = load_and_preprocess_dns_data()
    
    # 1. Run the heavy search
    best_features_dict = run_tournament(X_train, y_train, X_val, y_val)
    best_features_dict["All Features"] = list(range(X_train.shape[1]))

    # 2. Persist discovered indices so the notebook can load them dynamically
    import json
    indices_path = f"{OUT_DIR}/best_feature_indices.json"
    with open(indices_path, "w") as f:
        json.dump(best_features_dict, f, indent=2)
    log_print(f"\n💾 Saved best feature indices to `{indices_path}`")

    log_print("\n\n📊 Running 10-Fold Cross Validation and Statistical P-Tests on FULL Dataset...")
    X_full = np.concatenate([X_train, X_val, X_test], axis=0) # Total 100% of the dataset
    y_full = np.concatenate([y_train, y_val, y_test], axis=0)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    kfold_results = {}
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)

    for algo, indices in best_features_dict.items():
        log_print(f"  -> K-Fold for {algo} ({len(indices)} features)...")
        if len(indices) == 0:
            kfold_results[algo] = np.zeros(10)
            continue
        X_sub = X_full[:, indices]
        scores = cross_val_score(clf, X_sub, y_full, cv=skf, scoring='f1_weighted', n_jobs=-1)
        kfold_results[algo] = scores
        log_print(f"     Mean F1: {scores.mean():.4f} +/- {scores.std():.4f}")

    log_print("\n🔬 Performing Paired T-Tests (Statistical Significance)...")
    p_test_results = []
    prop_scores = kfold_results["Proposed-Penalty"]

    for algo, scores in kfold_results.items():
        if algo == "Proposed-Penalty": continue
        t_stat, p_val = ttest_rel(prop_scores, scores)
        sig = "Yes (p < 0.05)" if p_val < 0.05 else "No"
        better = "Proposed" if prop_scores.mean() > scores.mean() else algo
        
        p_test_results.append({
            "Comparison": f"Proposed vs {algo}",
            "P-Value": f"{p_val:.4e}",
            "T-Statistic": f"{t_stat:.4f}",
            "Significant?": sig,
            "Winner": better
        })

    df_p_tests = pd.DataFrame(p_test_results)
    df_p_tests.to_csv(f"{OUT_DIR}/statistical_p_tests.csv", index=False)
    log_print("\n--- P-Test Results ---")
    log_print(df_p_tests.to_string(index=False))

    # Boxplot for K-Fold
    df_box = pd.DataFrame(kfold_results)
    df_box.to_csv(f"{OUT_DIR}/kfold_raw_scores.csv", index=False)
    
    # Needs explicit conversion for seaborn melt without warnings
    df_box_melted = df_box.melt(var_name="Algorithm", value_name="F1_Score")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Algorithm", y="F1_Score", data=df_box_melted, palette="Set2")
    sns.stripplot(x="Algorithm", y="F1_Score", data=df_box_melted, color=".25", jitter=True)
    plt.title("10-Fold Cross-Validation Performance Across Algorithms", fontsize=14)
    plt.ylabel("F1-Weighted Score")
    plt.savefig(f"{OUT_DIR}/kfold_boxplot.png", dpi=300)

    log_print(f"\n✅ Heavy run complete! Output saved to {OUT_DIR}/")

if __name__ == "__main__":
    open(LOG_FILE, 'w').close()
    main()
