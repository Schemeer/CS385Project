# 需要使用 sklearn 的 metrics 库
from sklearn import metrics

# 计算 AUC 的函数，注意: average参数要设置成'macro'
metrics.roc_auc_score()

# 计算 macro F1 的函数，注意: average参数要设置成'macro'
metrics.f1_score()

# 计算 macro F1 的函数，注意: average要设置成'micro'
metrics.f1_score()


# BaseLine的指标:
#    AUC: 93.82;
#    Macro F1: 57.18
#    Micro F1: 73.37