import pandas as pd
import numpy as np
from loguru import logger
df = pd.read_csv("D://yeyuxiang_workdir//data//lidc_data//feature_engineering//radiomics_features.csv")
df = df.loc[df.malignancy_label != 3]
df.malignancy_label = df.malignancy_label.map(lambda x: 1 if x > 3 else 0)

X = df.values[:, 1:-1]
y = df.values[:, -1]

feature_names = df.columns[1:-1].values.squeeze().tolist()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler()
scale.fit(X)
X = scale.transform(X)
y = y.astype(np.uint8)



"""
evaluation
"""
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
import numpy as np


roc_display_list = []
pr_display_list = []


def pr_roc_plot(y_test, y_score, name):
    prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=1)
    roc_display_list.append(PrecisionRecallDisplay(precision=prec, recall=recall, estimator_name=name))
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    pr_display_list.append(RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=name + "-%.4f" % auc))


"""
LogisticRegression
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
coefs = clf.coef_

sorted_coefs = sorted(zip(feature_names, coefs[0]), key=lambda x: x[1], reverse=True)
for k, v in sorted_coefs[:20]:
    logger.critical("Top5 features:{} {}".format(k, v))
y_pred = clf.predict_proba(X_test)[:, 1]
pr_roc_plot(y_test, y_pred, name='logisticReg-AUC:')

"""
Lasso Regression
"""
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.001)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
pr_roc_plot(y_test, y_pred, name='Lasso-AUC:')


"""
BayesianRidge Regression
"""
clf = linear_model.BayesianRidge()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
pr_roc_plot(y_test, y_pred, name='BayesianRidge-AUC:')


"""
svc
"""

from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
pr_roc_plot(y_test, y_pred, name='SVC-AUC:')

"""
tree
"""
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
pr_roc_plot(y_test, y_pred, name='decisionTree-AUC:')


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
pr_roc_plot(y_test, y_pred, name='RF-AUC:')


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
y_pred = clf.predict(X_test)
pr_roc_plot(y_test, y_pred, name='mlp-AUC:')


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X, y)
y_pred = clf.predict(X_test)
pr_roc_plot(y_test, y_pred, name='Adaboost-AUC:')




fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
for roc_dis, pr_dis in zip(roc_display_list, pr_display_list):
    roc_dis.plot(ax=ax1)
    pr_dis.plot(ax=ax2)

my_x_ticks = np.arange(0, 1.05, 0.1)
my_y_ticks = np.arange(0, 1.05, 0.1)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)


plt.grid()
plt.legend()
plt.show()