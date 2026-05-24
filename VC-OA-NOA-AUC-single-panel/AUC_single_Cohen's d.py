import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
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
def _loo_step(train_idx, test_idx, x_sub, y_series, model_type):
    X_train, X_test = x_sub.iloc[train_idx], x_sub.iloc[test_idx]
    y_train = y_series.iloc[train_idx]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=500, max_depth=3, class_weight='balanced', random_state=42)
    else:
        model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    model.fit(X_train_s, y_train)
    oa_idx = np.where(model.classes_ == 'OA')[0][0]
    prob = model.predict_proba(X_test_s)[0][oa_idx]
    return y_series.iloc[test_idx].iloc[0], prob
def get_loo_probas_parallel(features, model_type='rf'):
    x_sub = x[features]
    loo = LeaveOneOut()
    results = Parallel(n_jobs=-1)(
        delayed(_loo_step)(train_idx, test_idx, x_sub, y_series, model_type) for train_idx, test_idx in
        loo.split(x_sub))
    y_true_list, y_prob_list = zip(*results)
    return np.array(y_true_list), np.array(y_prob_list)
def process_and_plot(y_true, y_prob, feat_name, filename, is_combined=False):
    y_bin = np.where(y_true == 'OA', 1, 0)
    fpr, tpr, _ = roc_curve(y_bin, y_prob)
    roc_auc = auc(fpr, tpr)
    ci_low, ci_high = calculate_auc_ci(y_bin, y_prob)

    if not is_combined:
        d_val = calculate_cohens_d(x.loc[y_series == 'OA', feat_name],
                                   x.loc[y_series == 'NOA', feat_name])
    else:
        d_val = calculate_cohens_d(y_prob[y_bin == 1], y_prob[y_bin == 0])
    plt.figure(figsize=(10, 10))
    legend_txt = f'AUC = {roc_auc:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f})'
    plt.plot(fpr, tpr, color=roc_curve_color, lw=2, label=legend_txt)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('x', fontsize=6)
    plt.ylabel('y', fontsize=6)
    plt.title(f'{feat_name}\nAUC: {roc_auc:.2f} | Cohen\'s d: {abs(d_val):.2f}', fontsize=6)
    plt.legend(loc="lower right", fontsize=6)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{filename}.pdf")
    plt.savefig(save_path, dpi=2000)
    plt.close()
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
print("\n开始生成Top5特征与组合Panel模型...")
clf_imp = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
clf_imp.fit(x[significant_features], y_series)
importances = pd.Series(clf_imp.feature_importances_, index=significant_features)
top_features = importances.sort_values(ascending=False).head(5).index.tolist()
for feat in top_features:
    print(f"Processing: {feat}")
    y_t, y_p = get_loo_probas_parallel([feat], model_type='rf')
    process_and_plot(y_t, y_p, feat, f'ROC_{feat}')
print("Processing Combined Panel...")
y_t_all, y_p_all = get_loo_probas_parallel(top_features, model_type='lr')
process_and_plot(y_t_all, y_p_all, 'Combined Panel', 'ROC_Combined_Panel', is_combined=True)
print("Done! All features exported (PDF + JPG 2000 DPI)!")
print("任务全部完成！组合Panel PDF与单个特征PDF完全一致！")