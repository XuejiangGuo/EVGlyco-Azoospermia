import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFECV

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
roc_curve_color = '#008000'
save_dir = './fig/individual_features'
os.makedirs(save_dir, exist_ok=True)
rawdata = pd.read_csv('Diffanalysis(p.adjust0.05_fc2).csv', index_col=0)
significant_features = rawdata[rawdata['Diff'] == 'YES'].index.tolist()
an_col = pd.read_csv('an_col.csv')
an_col = an_col[an_col['Type'].isin(['OA', 'HS', 'MA', 'SCO'])]
an_col.loc[an_col['Type'].isin(['HS', 'MA', 'SCO']), 'Type'] = 'NOA'
x = rawdata[an_col['Sample_id']].T
y_series = an_col.set_index('Sample_id')['Type']
y_bin = np.where(y_series == 'OA', 1, 0)
# RFECV feature selection
clf_rf = RandomForestClassifier(n_jobs=1, random_state=42)
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rfecv = RFECV(
    estimator=clf_rf,
    step=1,
    cv=cv_strat,
    scoring='roc_auc',
    n_jobs=1
)
rfecv.fit(x[significant_features], y_bin)
selected_features = np.array(significant_features)[rfecv.support_].tolist()

def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_sd
def calculate_auc_ci(y_true_bin, y_prob, n_bootstraps=1000):
    boot_aucs = []
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true_bin))
    for _ in range(n_bootstraps):
        resampled_idx = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(y_true_bin[resampled_idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[resampled_idx], y_prob[resampled_idx])
        boot_aucs.append(auc(fpr, tpr))
    sorted_aucs = np.sort(boot_aucs)
    if len(sorted_aucs) == 0:
        return 0, 0
    return sorted_aucs[int(0.025 * len(sorted_aucs))], sorted_aucs[int(0.975 * len(sorted_aucs))]

def get_loo_probas(features, x_data, y_labels):
    x_sub = x_data[features]
    y_true_list = []
    y_prob_list = []
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(x_sub):
        clf = RandomForestClassifier(n_jobs=1, random_state=42)
        clf.fit(x_sub.iloc[train_idx], y_labels.iloc[train_idx])
        oa_idx = np.where(clf.classes_ == 'OA')[0][0]
        prob = clf.predict_proba(x_sub.iloc[test_idx])[0][oa_idx]
        y_true_list.append(y_labels.iloc[test_idx].iloc[0])
        y_prob_list.append(prob)
    return np.array(y_true_list), np.array(y_prob_list)
def clean_filename(name):
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        name = name.replace(char, '_')
    return name
exported_features = set()
idx_oa = y_series[y_series == 'OA'].index
idx_noa = y_series[y_series == 'NOA'].index

for feat in significant_features:
    y_t, y_p = get_loo_probas([feat], x, y_series)
    y_bin_current = np.where(y_t == 'OA', 1, 0)
    fpr, tpr, _ = roc_curve(y_bin_current, y_p)
    roc_auc = auc(fpr, tpr)
    ci_low, ci_high = calculate_auc_ci(y_bin_current, y_p)
    feat_vals = x[feat]
    val_oa = feat_vals.loc[idx_oa]
    val_noa = feat_vals.loc[idx_noa]
    d_val = calculate_cohens_d(val_oa, val_noa)
    plt.figure(figsize=(6, 6))
    legend_txt = f'AUC = {roc_auc:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f})'
    plt.plot(fpr, tpr, color=roc_curve_color, lw=1.5, label=legend_txt)
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('x', fontsize=6)
    plt.ylabel('y', fontsize=6)
    plt.title(f'{feat}\nAUC: {roc_auc:.2f} | Cohen\'s d: {d_val:.2f}', fontsize=6)
    plt.legend(loc="lower right", fontsize=6)
    plt.tight_layout()
    safe_name = clean_filename(feat)
    save_path_pdf = os.path.join(save_dir, f"{safe_name}.pdf")
    save_path_jpg = os.path.join(save_dir, f"{safe_name}.jpg")
    plt.savefig(save_path_pdf, dpi=2000)
    plt.savefig(save_path_jpg, dpi=2000)
    plt.close()

print("Done! All features exported (PDF + JPG 2000 DPI)!")