import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, auc
if not os.path.exists('./fig'):
    os.makedirs('./fig')
RANDOM_SEED = 42
rawdata = pd.read_csv('Diffanalysis(p.adjust0.05_fc2).csv', index_col=0)
rawdata = rawdata[rawdata['Diff'] == 'YES']
an_col = pd.read_csv('an_col.csv')
an_col = an_col[an_col['Type'].isin(['OA', 'HS', 'MA', 'SCO'])]
an_col.loc[an_col['Type'].isin(['HS', 'MA', 'SCO']), 'Type'] = 'NOA'
an_col['Type_encoded'] = an_col['Type'].map({'OA': 1, 'NOA': 0})
x = rawdata[an_col['Sample_id']].T
y = an_col['Type_encoded']
feature_imp = pd.DataFrame(index=x.columns)
for i in range(20):
    clf = RandomForestClassifier(
        n_jobs=-1,
        random_state=RANDOM_SEED + i
    )
    clf.fit(x, y)
    feature_imp[f'RF_{i}'] = clf.feature_importances_
feature_imp['Mean_importance'] = feature_imp.mean(axis=1)
feature_imp['Std_importance'] = feature_imp.std(axis=1)
feature_imp['Rank_importance'] = feature_imp['Mean_importance'].rank(ascending=False)
feature_imp = feature_imp.sort_values(by='Rank_importance')
top = 20
top_features = feature_imp[feature_imp['Rank_importance'] <= top].index
x_top = x[top_features]
y_true = []
y_prob = []
loo = LeaveOneOut()
for train_index, test_index in loo.split(x_top):
    clf = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_SEED)
    clf.fit(x_top.iloc[train_index], y.iloc[train_index])
    y_true.append(y.iloc[test_index[0]])
    prob_oa = clf.predict_proba(x_top.iloc[test_index])[0, 1]
    y_prob.append(prob_oa)
y_true = np.array(y_true)
y_prob = np.array(y_prob)
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 9))
plt.plot(fpr, tpr, color='darkorange', lw=0.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
plt.xlim([0.0, 1.4])
plt.ylim([0.0, 1.4])
plt.xlabel('x', fontsize=8)
plt.ylabel('y', fontsize=8)
plt.title(f'Top {top} Features (Positive Class: OA=1)', fontsize=8)
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig('./fig/OA-NOA_AllImportanceProteins.pdf', dpi=2000)
plt.savefig('./fig/OA-NOA_AllImportanceProteins.jpg', dpi=2000)
plt.close()
clf_shap = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_SEED)
clf_shap.fit(x_top, y)
explainer = shap.Explainer(clf_shap)
shap_values = explainer(x_top)
plt.figure(figsize=(6, 9))
shap.summary_plot(shap_values[:, :, 1], x_top, max_display=40, show=False)
plt.title('SHAP Summary Plot (Top 20 Features, Positive Class: OA=1)', fontsize=9)
plt.tight_layout()
plt.savefig('./fig/OA-NOA_TopFeature_Shape.pdf', dpi=300)
plt.savefig('./fig/OA-NOA_TopFeature_Shape.jpg', dpi=2000)
plt.close()
print("Analysis completed!")