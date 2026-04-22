import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import local data loader
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(root_dir)
from utils.data_loader import load_and_preprocess_dns_data

def run_benchmark():
    print("--- Starting Final Comprehensive PhD Benchmark ---")
    print("Device: Apple M3 / RAM: 24GB")
    
    # 1. Load Full Data (1.1M Rows)
    print("Loading 1.1 million samples...")
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names = load_and_preprocess_dns_data()
    
    # DEV TEST MODE: Subsample if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("!!! RUNNING IN TEST MODE (1,000 samples only) !!!")
        X_train, y_train = X_train[:1000], y_train[:1000]
        X_test, y_test = X_test[:500], y_test[:500]
        
    print(f"Data Loaded. Features: {len(feature_names)}")
    
    # 2. Define Subsets
    # Indices from heavy_master_results/best_feature_indices.json
    best_indices = {
        "Proposed-Penalty": [0, 1, 4, 6, 9, 10, 11, 13, 22, 31],
        "Matrix-GA": [0, 1, 2, 4, 5, 6, 9, 10, 11, 12, 14, 15, 19, 23, 25, 26, 29, 30, 31],
        "JAYA-GA": [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 14, 17, 18, 19, 24, 25, 26, 30, 31],
        "Full-Set": list(range(len(feature_names)))
    }
    
    # 3. Define Models (The "Usual Suspects")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
        "KNN-Optimized": KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
        "SVM-RBF-Approx": make_pipeline(RBFSampler(gamma=1, random_state=42), LinearSVC(dual=False, random_state=42)),
        "Deep ANN": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42)
    }
    
    results_dir = "experiment_results/final_audit/separated"
    os.makedirs(results_dir, exist_ok=True)
    
    master_results = []
    
    # 4. Main Experimental Loop (Separated)
    for subset_name, indices in best_indices.items():
        print(f"\n--- Testing Subset: {subset_name} ({len(indices)} features) ---")
        X_train_sub = X_train[:, indices]
        X_test_sub = X_test[:, indices]
        
        for model_name, model in models.items():
            print(f"  > Training {model_name}...")
            start_time = time.time()
            
            # Train
            model.fit(X_train_sub, y_train)
            train_time = time.time() - start_time
            
            # Predict
            print(f"    Evaluating {model_name}...")
            y_pred = model.predict(X_test_sub)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            res = {
                "subset": subset_name,
                "model": model_name,
                "feature_count": len(indices),
                "accuracy": acc,
                "f1_score": f1,
                "precision": prec,
                "recall": rec,
                "mcc": mcc,
                "train_time_sec": train_time
            }
            master_results.append(res)
            
            # Save Separated JSON
            file_base = f"{subset_name.replace('-', '_')}_{model_name.replace(' ', '_')}"
            with open(f"{results_dir}/metrics_{file_base}.json", 'w') as f:
                json.dump(res, f, indent=4)
            
            # Save Confusion Matrix CSV
            np.savetxt(f"{results_dir}/cm_{file_base}.csv", cm, delimiter=",", fmt='%d')
            
            print(f"    Finished. F1: {f1:.4f} | Time: {train_time:.2f}s")

    # 5. Ablation Study (Phase 3)
    print("\n--- Starting Phase 3: Ablation Study (Proposed Subset) ---")
    prop_indices = best_indices["Proposed-Penalty"]
    X_train_prop = X_train[:, prop_indices]
    
    # 5.1 Identify Top Feature via RF
    temp_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    temp_rf.fit(X_train_prop, y_train)
    importances = temp_rf.feature_importances_
    top_idx_in_prop = np.argmax(importances)
    top_feature_global_idx = prop_indices[top_idx_in_prop]
    top_feature_name = feature_names[top_feature_global_idx]
    
    print(f"Top Feature identified for Ablation: {top_feature_name}")
    
    # 5.2 Remove it and re-test
    ablation_indices = [idx for idx in prop_indices if idx != top_feature_global_idx]
    X_train_abl = X_train[:, ablation_indices]
    X_test_abl = X_test[:, ablation_indices]
    
    for model_name, model in models.items():
        print(f"  > Ablation Test: {model_name} (Minus {top_feature_name})...")
        model.fit(X_train_abl, y_train)
        y_pred = model.predict(X_test_abl)
        f1_abl = f1_score(y_test, y_pred, average='weighted')
        
        # Find original score
        orig_score = next(item["f1_score"] for item in master_results if item["subset"] == "Proposed-Penalty" and item["model"] == model_name)
        delta = orig_score - f1_abl
        
        abl_res = {
            "subset": "Ablation (Minus Top-1)",
            "model": model_name,
            "feature_count": len(ablation_indices),
            "removed_feature": top_feature_name,
            "f1_score": f1_abl,
            "impact_delta": delta
        }
        master_results.append(abl_res)
        print(f"    F1 Drop: {delta:.4f}")

    # 6. Consolidation (Master Table)
    df_results = pd.DataFrame(master_results)
    df_results.to_csv("experiment_results/final_audit/master_appendix_results.csv", index=False)
    
    # 7. Generate Plotly Dashboard
    # Filter out ablation for the main chart
    df_main = df_results[df_results['subset'] != 'Ablation (Minus Top-1)']
    
    fig = px.bar(df_main, x="subset", y="f1_score", color="model", barmode="group",
                 title="Final Benchmarking Results: F1-Score Comparison",
                 labels={"f1_score": "Weighted F1-Score", "subset": "Feature Subset"})
    
    # Add Ablation chart
    df_abl = df_results[df_results['subset'] == 'Ablation (Minus Top-1)']
    fig_abl = px.bar(df_abl, x="model", y="impact_delta", 
                      title=f"Ablation Impact: Performance Drop without Top Feature ({top_feature_name})",
                      labels={"impact_delta": "F1-Score Drop (Criticality)"})
    
    # Save combined dashboard (simplified)
    with open("experiment_results/final_audit/final_dashboard.html", "w") as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig_abl.to_html(full_html=False, include_plotlyjs='cdn'))
    
    print("\n--- Benchmark Completed Successfully ---")

    print(f"Master CSV: experiment_results/final_audit/master_appendix_results.csv")
    print(f"Dashboard: experiment_results/final_audit/final_dashboard.html")

if __name__ == "__main__":
    run_benchmark()
