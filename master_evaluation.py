import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
from utils.data_loader import load_and_preprocess_dns_data
from models.rf_evaluator import RFEvaluator

# ==========================================
# CONSTANT CONFIGURATIONS
# ==========================================
OUT_DIR = "master_figures"
os.makedirs(OUT_DIR, exist_ok=True)
CLASS_NAMES = ['Benign', 'Iodine', 'DNS2TCP', 'Dnscat2']

# Based on previous optimal runs from the GAs
FEATURE_SETS = {
    "Baseline (All 32)": list(range(32)),
    "Proposed-Penalty GA (20)": [0, 2, 4, 5, 7, 8, 10, 11, 14, 15, 17, 18, 19, 21, 23, 24, 25, 27, 29, 31],
    "SOTA: Matrix-GA (13)": [0, 3, 5, 8, 11, 14, 17, 20, 22, 25, 28, 30, 31],
    "SOTA: JAYA-GA (17)": [1, 2, 4, 6, 8, 9, 12, 14, 15, 18, 20, 22, 24, 26, 28, 29, 31]
}

def main():
    print(f"🚀 Initializing Master Evaluation Run... Saving outputs to `{OUT_DIR}/`")
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names = load_and_preprocess_dns_data()
    
    # ---------------------------------------------------------
    # PART 1: COMPONENT ABLATION (Comparing Feature Sets)
    # ---------------------------------------------------------
    print("\n[1/5] Evaluating SOTA Feature Sets...")
    feature_results = []
    trained_models = {}

    for name, indices in FEATURE_SETS.items():
        print(f"  -> Training RF on {name}...")
        start_train = time.time()
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        clf.fit(X_train[:, indices], y_train)
        train_time = time.time() - start_train
        
        start_inf = time.time()
        y_pred = clf.predict(X_test[:, indices])
        inf_time = (time.time() - start_inf) / len(X_test) * 1000
        
        trained_models[name] = clf
        feature_results.append({
            "Feature Subset Strategy": name,
            "Feature Count": len(indices),
            "F1-Weighted": f1_score(y_test, y_pred, average="weighted"),
            "MCC Score": matthews_corrcoef(y_test, y_pred),
            "Train Time (s)": train_time,
            "Latency (ms/pkt)": inf_time
        })

    df_feats = pd.DataFrame(feature_results)
    df_feats.to_csv(f"{OUT_DIR}/table_feature_subset_comparison.csv", index=False)

    # 📈 Plot 1: Pareto Efficiency Scatter (Accuracy vs Feature Count)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_feats, x="Feature Count", y="F1-Weighted", hue="Feature Subset Strategy", s=300, palette="viridis")
    for i in range(df_feats.shape[0]):
        plt.text(df_feats["Feature Count"][i] + 0.3, df_feats["F1-Weighted"][i], df_feats["Feature Subset Strategy"][i], fontsize=10)
    plt.title("Ablation: The Impact of Feature Subsets on Detection Accuracy", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUT_DIR}/fig1_feature_ablation_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PART 2: CLASSIFIER BASELINE ABLATION (RF vs LR vs DT)
    # ---------------------------------------------------------
    print("\n[2/5] Evaluating Machine Learning Classifier Baselines...")
    classifiers = {
        "Random Forest (Proposed)": RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
        "Logistic Regression (LR)": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        "Decision Tree (DT)": DecisionTreeClassifier(max_depth=10, random_state=42),
    }

    baseline_results = []
    # Test all classifiers strictly on the Proposed-Penalty Feature subset
    prop_idx = FEATURE_SETS["Proposed-Penalty GA (20)"]
    
    for clf_name, clf in classifiers.items():
        print(f"  -> Testing {clf_name}...")
        clf.fit(X_train[:, prop_idx], y_train)
        y_pred = clf.predict(X_test[:, prop_idx])
        
        baseline_results.append({
            "Classifier": clf_name,
            "F1-Weighted": f1_score(y_test, y_pred, average="weighted"),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Acc": balanced_accuracy_score(y_test, y_pred)
        })

    df_base = pd.DataFrame(baseline_results)
    df_base.to_csv(f"{OUT_DIR}/table_classifier_ablation.csv", index=False)

    # 📈 Plot 2: Classifier Bar Chart
    df_melt = pd.melt(df_base, id_vars="Classifier", var_name="Metric", value_name="Score")
    plt.figure(figsize=(11, 6))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Classifier", palette="mako")
    plt.ylim(0.7, 1.0)
    plt.title("Classifier Head-to-Head using the Proposed Feature Subset (20 features)", fontsize=14)
    plt.legend(loc='lower left')
    plt.savefig(f"{OUT_DIR}/fig2_classifier_baseline_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PART 3: PROPOSED MODEL IN-DEPTH VISUALIZATIONS (ROC, PR, CM, FI)
    # ---------------------------------------------------------
    print("\n[3/5] Generating High-Resolution Model Visualizations...")
    best_clf = trained_models["Proposed-Penalty GA (20)"]
    y_score = best_clf.predict_proba(X_test[:, prop_idx])
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    y_pred = best_clf.predict(X_test[:, prop_idx])

    # 📈 Plot 3: Multi-Class ROC Curve
    plt.figure(figsize=(10, 7))
    colors = ['navy', 'darkorange', 'darkgreen', 'darkred']
    for i in range(4):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{CLASS_NAMES[i]} (AUC = {auc(fpr, tpr):.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('Multi-Class ROC Curve (Proposed GA-RF)', fontsize=15)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(f"{OUT_DIR}/fig3_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 📈 Plot 4: Precision-Recall Curve
    plt.figure(figsize=(10, 7))
    for i in range(4):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, color=colors[i], lw=2, label=f'{CLASS_NAMES[i]} (mAP = {ap:.4f})')
    plt.title('Multi-Class Precision-Recall Curve', fontsize=15)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(f"{OUT_DIR}/fig4_pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 📈 Plot 5: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (Proposed Framework)", fontsize=15)
    plt.ylabel("True Class"); plt.xlabel("Predicted Class")
    plt.savefig(f"{OUT_DIR}/fig5_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 📈 Plot 6: Feature Importance
    importances = best_clf.feature_importances_
    elite_features = [feature_names[i] for i in prop_idx]
    
    # Sort them
    indices = np.argsort(importances)[::-1]
    sorted_features = [elite_features[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=sorted_importances[:15], y=sorted_features[:15], palette="viridis")
    plt.title("Top 15 Most Important Network Features Selected by GA", fontsize=15)
    plt.xlabel("Gini Importance")
    plt.savefig(f"{OUT_DIR}/fig6_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PART 4: DATASET STARVATION (Robustness Study)
    # ---------------------------------------------------------
    print("\n[4/5] Running Dataset Starvation Robustness Tests...")
    scenarios = [
        ("Full Test Mix", [0, 1, 2, 3]),
        ("Missing Iodine", [0, 2, 3]),
        ("Missing DNS2TCP", [0, 1, 3]),
        ("Missing Dnscat2", [0, 1, 2])
    ]

    robustness_results = []
    for scenario_name, classes_kept in scenarios:
        mask = np.isin(y_test, classes_kept)
        y_pred_rob = best_clf.predict(X_test[mask][:, prop_idx])
        
        robustness_results.append({
            "Scenario": scenario_name,
            "Accuracy": accuracy_score(y_test[mask], y_pred_rob),
            "F1-Weighted": f1_score(y_test[mask], y_pred_rob, average="weighted")
        })

    df_rob = pd.DataFrame(robustness_results)
    df_rob.to_csv(f"{OUT_DIR}/table_dataset_robustness.csv", index=False)

    # 📈 Plot 7: Dataset Robustness Heatmap
    heatmap_data = df_rob.set_index("Scenario")[["Accuracy", "F1-Weighted"]]
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".4f", cbar=True)
    plt.title("Dataset Starvation: Model Robustness Across Attack Classes", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig7_data_robustness_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Master Evaluation Complete! Designed for IAENG Reviewers. All 7 High-Res images and CSVs saved to `{OUT_DIR}/`.")

if __name__ == "__main__":
    main()
