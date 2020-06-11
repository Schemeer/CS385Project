
from sklearn import metrics
import numpy as np

# y_test = np.array([0,0,1,1])
# y_pred1 = np.array([0.3,0.2,0.25,0.7])
# y_pred2 = np.array([0,0,1,0])
# 计算 AUC 的函数，注意: average参数要设置成'macro'
def auc(y_true,y_pred):
    auc = metrics.roc_auc_score(y_true,y_pred)
    return auc

def macro_f1(y_true,y_pred):
    f1 = metrics.f1_score(y_true = y_true ,y_pred = y_pred, average = "macro")
    return f1

def micro_f1(y_true,y_pred):
    f1 = metrics.f1_score(y_true = y_true ,y_pred = y_pred, average = "micro")
    return f1 


# print(macro_f1(y_test,y_pred2))
# print(micro_f1(y_test,y_pred2))


# BaseLine的指标:
#    AUC: 93.82;
#    Macro F1: 57.18
#    Micro F1: 73.37