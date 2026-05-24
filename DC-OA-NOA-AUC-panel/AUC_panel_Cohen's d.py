import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc

# --- 样式设置 ---
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 6
roc_curve_color = '#008000'

# 2. 数据加载
rawdata = pd.read_csv('Diffanalysis(p.adjust0.05_fc2).csv', index_col=0)
rawdata = rawdata[rawdata['Diff'] == 'YES']

an_col = pd.read_csv('an_col.csv')
an_col = an_col[an_col['Type'].isin(['OA', 'HS', 'MA', 'SCO'])]
an_col.loc[an_col['Type'].isin(['HS', 'MA', 'SCO']), 'Type'] = 'NOA'

x = rawdata[an_col['Sample_id']].T
y_series = an_col.set_index('Sample_id')['Type']


# --- 计算函数 ---
def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_sd


def calculate_auc_ci(y_true_bin, y_prob, n_bootstraps=1000):
    bootstrapped_aucs = []
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true_bin))
    for i in range(n_bootstraps):
        resampled_indices = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(y_true_bin[resampled_indices])) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(y_true_bin[resampled_indices], y_prob[resampled_indices])
        bootstrapped_aucs.append(auc(fpr, tpr))

    sorted_aucs = np.sort(bootstrapped_aucs)
    conf_lower = sorted_aucs[int(0.025 * len(sorted_aucs))] if len(sorted_aucs) > 0 else 0.0
    conf_upper = sorted_aucs[int(0.975 * len(sorted_aucs))] if len(sorted_aucs) > 0 else 0.0
    return conf_lower, conf_upper


# 4. 留一法交叉验证
def get_loo_probas(features):
    x_sub = x[features]
    y_true_list = []
    y_prob_list = []
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(x_sub):
        clf = RandomForestClassifier(n_jobs=1, random_state=42)
        clf.fit(x_sub.iloc[train_idx], y_series.iloc[train_idx])
        y_true_list.append(y_series.iloc[test_idx].iloc[0])
        oa_idx = np.where(clf.classes_ == 'OA')[0][0]
        prob = clf.predict_proba(x_sub.iloc[test_idx])[0][oa_idx]
        y_prob_list.append(prob)
    return np.array(y_true_list), np.array(y_prob_list)


# 文件名清洗
def clean_filename(name):
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        name = name.replace(char, '_')
    return name


# 5. 绘图逻辑
os.makedirs('./fig/individual_features', exist_ok=True)


def plot_and_save_roc(y_true, y_prob, display_name, filename, is_combined=False, feature_name=None):
    y_bin = np.where(y_true == 'OA', 1, 0)
    fpr, tpr, _ = roc_curve(y_bin, y_prob)
    roc_auc = auc(fpr, tpr)
    ci_low, ci_high = calculate_auc_ci(y_bin, y_prob)

    if not is_combined and feature_name:
        val_oa = x.loc[y_series == 'OA', feature_name]
        val_noa = x.loc[y_series == 'NOA', feature_name]
    else:
        val_oa = y_prob[y_true == 'OA']
        val_noa = y_prob[y_true == 'NOA']

    d_val = calculate_cohens_d(val_oa, val_noa)

    plt.figure(figsize=(9, 9))
    legend_label = f'AUC = {roc_auc:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f})'

    plt.plot(fpr, tpr, color=roc_curve_color, lw=1.5, label=legend_label)
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('x')
    plt.ylabel('y')

    plt.title(f"{display_name}\nCohen's d: {d_val:.2f}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # 已删除 bbox_in='tight'
    plt.savefig(f'./fig/individual_features/{filename}.pdf', dpi=2000)
    plt.savefig(f'./fig/individual_features/{filename}.jpg', dpi=2000)
    plt.close()


# ===================== 执行：所有特征全部导出 =====================
all_features = x.columns.tolist()

for feat in all_features:
    y_t, y_p = get_loo_probas([feat])
    safe_name = clean_filename(feat)
    plot_and_save_roc(y_t, y_p, feat, safe_name, is_combined=False, feature_name=feat)

print("Done! All features exported!")