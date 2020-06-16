from sklearn import metrics
import numpy as np
import torch

# y_test = np.array([0,0,1,1])
# y_pred1 = np.array([0.3,0.2,0.25,0.7])
# y_pred2 = np.array([0,0,1,0])
# 计算 AUC 的函数，注意: average参数要设置成'macro'
def auc(y_true, y_pred):
    auc = metrics.roc_auc_score(y_true, y_pred)
    return auc


def macro_f1(y_true,y_pred):
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    return f1


def micro_f1(y_true,y_pred):
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    return f1 


def report(y_true, y_pred):
    return metrics.classification_report(y_true, y_pred)
# print(macro_f1(y_test,y_pred2))
# print(micro_f1(y_test,y_pred2))


def three_scores(y_true,y_pred):
    return auc(y_true,y_pred), macro_f1(y_true,y_pred), micro_f1(y_true,y_pred)


def calculate_acuracy_mode_one(labels, model_pred):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    # 模型预测的结果中有多少个是正确的
    precision = true_predict_num / pred_one_num
    # 模型预测正确的结果中，占所有真实标签的数量
    recall = true_predict_num / target_one_num

    return precision.item(), recall.item()

# BaseLine的指标:
#    AUC: 93.82;
#    Macro F1: 57.18
#    Micro F1: 73.37