import os
import pandas as pd
import numpy as np
import matplotlib
import xgboost as xgb
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
INPUT_FILE = 'Diffanalysis(p.adjust0.05_fc2).csv'
SAVE_DIR = './fig'
RANDOM_STATE = 42
CV_FOLDS = 5
MAX_SELECTED_FEATURES = 10
BOOTSTRAP_TIMES = 1000
NATURE_COLORS = ['#37A2E9', '#66D0AA', '#DAACEC', '#FF61C3', '#6A8CAF', '#B5D6A0', '#F4B350', '#256D85']
FONT_FAMILY = 'Times New Roman'
AXES_LABEL_SIZE = 5
TICK_LABEL_SIZE = 5
LEGEND_FONT_SIZE = 5
AXES_LINE_WIDTH = 0.8
MAX_ITER = 2000
HIDDEN_LAYERS_SIMPLE = (5,)
HIDDEN_LAYERS_COMPLEX = (64, 32)
MLP_MAX_ITER = 3000
KNN_NEIGHBORS = 5

plt.rcParams['font.sans-serif'] = FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.labelsize"] = AXES_LABEL_SIZE
plt.rcParams["xtick.labelsize"] = TICK_LABEL_SIZE
plt.rcParams["ytick.labelsize"] = TICK_LABEL_SIZE
plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE
plt.rcParams["legend.frameon"] = True
plt.rcParams["axes.linewidth"] = AXES_LINE_WIDTH

def calculate_cohens_d(x_data, y_labels):
    group0 = x_data[y_labels == 0]
    group1 = x_data[y_labels == 1]
    n0, n1 = len(group0), len(group1)
    if n0 < 2 or n1 < 2: return 0
    v0, v1 = np.var(group0, ddof=1), np.var(group1, ddof=1)
    pooled_sd = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2))
    if pooled_sd == 0 or np.isnan(pooled_sd): return 0
    return (np.mean(group1) - np.mean(group0)) / pooled_sd

def calculate_auc_ci(y_true, y_prob):
    bootstrapped_aucs = []
    rng = np.random.RandomState(RANDOM_STATE)
    indices = np.arange(len(y_true))
    for _ in range(BOOTSTRAP_TIMES):
        resampled_idx = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(y_true[resampled_idx])) < 2: continue
        fpr, tpr, _ = roc_curve(y_true[resampled_idx], y_prob[resampled_idx])
        bootstrapped_aucs.append(auc(fpr, tpr))
    sorted_aucs = np.sort(bootstrapped_aucs)
    if len(sorted_aucs) == 0: return 0.0, 0.0
    return sorted_aucs[int(0.025 * len(sorted_aucs))], sorted_aucs[int(0.975 * len(sorted_aucs))]

class KNNWithImportance(KNeighborsClassifier):
    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        super().fit(X_arr, y)
        self.feature_importances_ = []
        for i in range(X_arr.shape[1]):
            s = cross_val_score(KNeighborsClassifier(n_neighbors=self.n_neighbors), X_arr[:, [i]], y, cv=3, scoring='roc_auc').mean()
            self.feature_importances_.append(s)
        self.feature_importances_ = np.array(self.feature_importances_) / (np.sum(self.feature_importances_) + 1e-9)
        return self

class GaussianNBWithImportance(GaussianNB):
    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        super().fit(X_arr, y)
        self.feature_importances_ = []
        for i in range(X_arr.shape[1]):
            s = cross_val_score(GaussianNB(), X_arr[:, [i]], y, cv=3, scoring='roc_auc').mean()
            self.feature_importances_.append(s)
        self.feature_importances_ = np.array(self.feature_importances_) / (np.sum(self.feature_importances_) + 1e-9)
        return self

class MLPWithImportance(MLPClassifier):
    def fit(self, X, y):
        super().fit(X, y)
        self.feature_importances_ = np.absolute(self.coefs_[0]).sum(axis=1)
        self.feature_importances_ /= (np.sum(self.feature_importances_) + 1e-9)
        return self

data = pd.read_csv(INPUT_FILE)
X = data.iloc[:, 2:]
y = data['Type'].map({'AZS_C1': 0, 'AZS_C2': 1}).astype(int)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

models = [
    {'name': 'LR', 'full_name': 'Logistic Regression', 'classifier': LogisticRegression(max_iter=MAX_ITER, class_weight='balanced', random_state=RANDOM_STATE), 'color': NATURE_COLORS[0], 'importance_getter': 'coef_'},
    {'name': 'KNN', 'full_name': 'K-Nearest Neighbors', 'classifier': KNNWithImportance(n_neighbors=KNN_NEIGHBORS), 'color': NATURE_COLORS[1], 'importance_getter': 'feature_importances_'},
    {'name': 'NBayes', 'full_name': 'Naive Bayes', 'classifier': GaussianNBWithImportance(), 'color': NATURE_COLORS[2], 'importance_getter': 'feature_importances_'},
    {'name': 'LDA', 'full_name': 'Linear Discriminant Analysis', 'classifier': LinearDiscriminantAnalysis(), 'color': NATURE_COLORS[3], 'importance_getter': 'coef_'},
    {'name': 'XGB', 'full_name': 'XGBoost', 'classifier': xgb.XGBClassifier(eval_metric='auc', random_state=RANDOM_STATE), 'color': NATURE_COLORS[4], 'importance_getter': 'feature_importances_'},
    {'name': 'DT', 'full_name': 'Decision Tree', 'classifier': DecisionTreeClassifier(random_state=RANDOM_STATE), 'color': NATURE_COLORS[5], 'importance_getter': 'feature_importances_'},
    {'name': 'NN', 'full_name': 'Neural Network', 'classifier': MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS_SIMPLE, max_iter=MAX_ITER, random_state=RANDOM_STATE), 'color': NATURE_COLORS[6], 'importance_getter': lambda clf: np.absolute(clf.coefs_[0]).sum(axis=1)},
    {'name': 'MLP', 'full_name': 'Multi-Layer Perceptron', 'classifier': MLPWithImportance(hidden_layer_sizes=HIDDEN_LAYERS_COMPLEX, max_iter=MLP_MAX_ITER, random_state=RANDOM_STATE), 'color': NATURE_COLORS[7], 'importance_getter': 'feature_importances_'}
]

panel_results = []
os.makedirs(SAVE_DIR, exist_ok=True)

for m in models:
    selector = RFECV(m['classifier'], step=1, cv=cv, scoring='roc_auc', min_features_to_select=1, importance_getter=m['importance_getter'], n_jobs=-1)
    selector.fit(X_scaled, y)
    support_idx = np.where(selector.support_)[0]
    if len(support_idx) > MAX_SELECTED_FEATURES:
        support_idx = support_idx[:MAX_SELECTED_FEATURES]
    X_selected = X_scaled.iloc[:, support_idx]
    y_probs = cross_val_predict(m['classifier'], X_selected, y, cv=cv, method='predict_proba')[:, 1]
    auc_val = roc_auc_score(y, y_probs)
    ci_low, ci_high = calculate_auc_ci(y, y_probs)
    panel_results.append({'name': m['name'], 'probs': y_probs, 'auc': auc_val, 'ci_low': ci_low, 'ci_high': ci_high, 'color': m['color']})

plt.figure(figsize=(6, 6))
for res in panel_results:
    fpr, tpr, _ = roc_curve(y, res['probs'])
    label_txt = f"{res['name']} (AUC: {res['auc']:.2f}, 95% CI: {res['ci_low']:.2f}-{res['ci_high']:.2f})"
    plt.plot(fpr, tpr, color=res['color'], lw=1, label=label_txt)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'ROC_Comparison_NoSVM_CI_F1.pdf'), dpi=2000)
plt.close()