import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc
from joblib import Parallel, delayed
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 7
save_dir = './fig/individual_features'
os.makedirs(save_dir, exist_ok=True)
rawdata = pd.read_csv('Diffanalysis(p.adjust0.05_fc2).csv', index_col=0)
significant_features = rawdata[rawdata['Diff'] == 'YES'].index.tolist()
an_col = pd.read_csv('an_col.csv')
an_col = an_col[an_col['Type'].isin(['HS', 'MA', 'SCO'])]
x = rawdata[an_col['Sample_id']].T
y_series = an_col.set_index('Sample_id')['Type']
class_names = np.sort(np.unique(y_series))
class_map = {'HS': 'HS subgroups', 'MA': 'MA subgroups', 'SCO': 'SCO subgroups'}
custom_colors = ['#FE8883', '#37A2E9', '#66D0AA']
def calculate_cohens_d_ovr(feature_values, target_y, target_class):
    group1 = feature_values.loc[target_y == target_class]
    group2 = feature_values.loc[target_y != target_class]
    n1, n2 = len(group1), len(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0 or np.isnan(pooled_sd): return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_sd
def calculate_multiclass_auc_ci(y_true, y_probs, classes, n_bootstraps=1000):
    boot_results = {c: [] for c in classes}
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    for _ in range(n_bootstraps):
        resampled_idx = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(y_true[resampled_idx])) < 2: continue
        for i, class_name in enumerate(classes):
            y_bin = np.where(y_true[resampled_idx] == class_name, 1, 0)
            if len(np.unique(y_bin)) < 2: continue
            fpr, tpr, _ = roc_curve(y_bin, y_probs[resampled_idx, i])
            boot_results[class_name].append(auc(fpr, tpr))
    return {c: (np.percentile(scores, 2.5), np.percentile(scores, 97.5))
    if len(scores) > 0 else (0.0, 0.0) for c, scores in boot_results.items()}
def run_single_loo_step(train_idx, test_idx, x_sub, y_labels):
    clf = RandomForestClassifier(n_jobs=1, random_state=42)
    clf.fit(x_sub.iloc[train_idx], y_labels.iloc[train_idx])
    return y_labels.iloc[test_idx].iloc[0], clf.predict_proba(x_sub.iloc[test_idx])[0]
def get_loo_probas(features):
    x_sub = x[features]
    results = Parallel(n_jobs=-1)(
        delayed(run_single_loo_step)(train_idx, test_idx, x_sub, y_series)
        for train_idx, test_idx in LeaveOneOut().split(x_sub)
    )
    y_true, y_probs = zip(*results)
    return np.array(y_true), np.array(y_probs)
def plot_multiclass_roc(y_true, y_probs, ci_dict, title, save_name, mean_d=None):
    plt.figure(figsize=(6, 6))
    for i, c in enumerate(class_names):
        y_bin = np.where(y_true == c, 1, 0)
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        low, high = ci_dict[c]
        plt.plot(fpr, tpr, color=custom_colors[i], lw=1.5,
                 label=f"{class_map[c]}\nAUC: {roc_auc:.2f} ({low:.2f}-{high:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    full_title = f"{title}\nMean Cohen's d: {mean_d:.2f}" if mean_d else title
    plt.title(full_title, fontsize=7)
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()

    plt.savefig(save_name.replace(".pdf", ".jpg"), dpi=2000)
    plt.savefig(save_name, dpi=2000)
    plt.close()
print(f"开始分析 {len(significant_features)} 个特征...")
for feat in significant_features:
    y_true, y_probs = get_loo_probas([feat])
    ci_dict = calculate_multiclass_auc_ci(y_true, y_probs, class_names)
    d_dict = {c: calculate_cohens_d_ovr(x[feat], y_series, c) for c in class_names}
    mean_d = np.mean([abs(v) for v in d_dict.values()])
    safe_name = feat.replace('/', '_').replace(':', '_')
    plot_multiclass_roc(y_true, y_probs, ci_dict, feat,
                        f"{save_dir}/NOAsub_{safe_name}.pdf", mean_d)
    print(f"已导出: {feat}")
if len(significant_features) >= 2:
    top_2 = significant_features[:2]
    print(f"分析 Panel (联合特征): {top_2}")
    y_true_p, y_probs_p = get_loo_probas(top_2)
    ci_dict_p = calculate_multiclass_auc_ci(y_true_p, y_probs_p, class_names)

    plot_multiclass_roc(y_true_p, y_probs_p, ci_dict_p,
                        f"Panel: {top_2[0]} + {top_2[1]}",
                        f"{save_dir}/NOAsub_Panel_Combined.pdf")

print("\n[完成] 所有特征已全部导出 PDF + JPG！")