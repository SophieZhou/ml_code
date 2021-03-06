# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:14:36 2022

@author: Administrator
"""

# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

def getDataSet():
    """
    西瓜数据集3.0alpha。 列：[密度，含糖量，好瓜]
    :return: np数组。
    """
    dataSet = [
        [0.697, 0.460, '是'],
        [0.774, 0.376, '是'],
        [0.634, 0.264, '是'],
        [0.608, 0.318, '是'],
        [0.556, 0.215, '是'],
        [0.403, 0.237, '是'],
        [0.481, 0.149, '是'],
        [0.437, 0.211, '是'],
        [0.666, 0.091, '否'],
        [0.243, 0.267, '否'],
        [0.245, 0.057, '否'],
        [0.343, 0.099, '否'],
        [0.639, 0.161, '否'],
        [0.657, 0.198, '否'],
        [0.360, 0.370, '否'],
        [0.593, 0.042, '否'],
        [0.719, 0.103, '否']
    ]

    # # '是'为1，'否'为-1
    for i in range(len(dataSet)):  
        if dataSet[i][-1] == '是':
            dataSet[i][-1] = 1
        else:
            dataSet[i][-1] = -1

    return np.array(dataSet)


def calErr(dataSet, feature, threshVal, inequal, D):
    """
    计算数据带权值的错误率。
    :param dataSet:     [密度，含糖量，好瓜]
    :param feature:     [密度，含糖量]
    :param threshVal:
    :param inequal:     'lt' or 'gt. (大于或小于）
    :param D:           数据的权重。错误分类的数据权重会大。
    :return:            错误率。
    """
    DFlatten = D.flatten()   
    errCnt = 0
    i = 0
    if inequal == 'lt':
        for data in dataSet:
            if (data[feature] <= threshVal and data[-1] == -1) or \
               (data[feature] > threshVal and data[-1] == 1):
                errCnt += 1 * DFlatten[i]
            i += 1
    else:
        for data in dataSet:
            if (data[feature] >= threshVal and data[-1] == -1) or \
               (data[feature] < threshVal and data[-1] == 1):
                errCnt += 1 * DFlatten[i]
            i += 1
    return errCnt


def buildStump(dataSet, D):
    """
    通过带权重的数据，建立错误率最小的决策树桩。
    :param dataSet:
    :param D:
    :return:    包含建立好的决策树桩的信息。如feature，threshVal，inequal，err。
    """
    m, n = dataSet.shape
    bestErr = np.inf
    bestStump = {}
    for i in range(n-1):                    # 对第i个特征
        for j in range(m):                  # 对第j个数据
            threVal = dataSet[j][i]
            for inequal in ['lt', 'gt']:    # 对于大于或等于符号划分。
                err = calErr(dataSet, i, threVal, inequal, D)  # 错误率
                if err < bestErr:           # 如果错误更低，保存划分信息。
                    bestErr = err
                    bestStump["feature"] = i
                    bestStump["threshVal"] = threVal
                    bestStump["inequal"] = inequal
                    bestStump["err"] = err

    return bestStump


def predict(data, bestStump):
    """
    通过决策树桩预测数据
    :param data:        待预测数据
    :param bestStump:   决策树桩。
    :return:
    """
    if bestStump["inequal"] == 'lt':
        if data[bestStump["feature"]] <= bestStump["threshVal"]:
            return 1
        else:
            return -1
    else:
        if data[bestStump["feature"]] >= bestStump["threshVal"]:
            return 1
        else:
            return -1


def AdaBoost(dataSet, T):
    """
    每学到一个学习器，根据其错误率确定两件事。
    1.确定该学习器在总学习器中的权重。正确率越高，权重越大。
    2.调整训练样本的权重。被该学习器误分类的数据提高权重，正确的降低权重，
      目的是在下一轮中重点关注被误分的数据，以得到更好的效果。
    :param dataSet:  数据集。
    :param T:        迭代次数，即训练多少个分类器
    :return:         字典，包含了T个分类器。
    """
    m, n = dataSet.shape
    D = np.ones((1, m)) / m                      # 初始化权重，每个样本的初始权重是相同的。
    classLabel = dataSet[:, -1].reshape(1, -1)   # 数据的类标签。
    G = {}      # 保存分类器的字典，

    for t in range(T):
        stump = buildStump(dataSet, D)           # 根据样本权重D建立一个决策树桩
        err = stump["err"]
        alpha = np.log((1 - err) / err) / 2      # 第t个分类器的权值
        # 更新训练数据集的权值分布
        pre = np.zeros((1, m))
        for i in range(m):
            pre[0][i] = predict(dataSet[i], stump)
        a = np.exp(-alpha * classLabel * pre)
        D = D * a / np.dot(D, a.T)

        G[t] = {}
        G[t]["alpha"] = alpha
        G[t]["stump"] = stump
    return G


def adaPredic(data, G):
    """
    通过Adaboost得到的总的分类器来进行分类。
    :param data:    待分类数据。
    :param G:       字典，包含了多个决策树桩
    :return:        预测值
    """
    score = 0
    for key in G.keys():
        pre = predict(data, G[key]["stump"])
        score += G[key]["alpha"] * pre
    flag = 0
    if score > 0:
        flag = 1
    else:
        flag = -1
    return flag


def calcAcc(dataSet, G):
    """
    计算准确度
    :param dataSet:     数据集
    :param G:           字典，包含了多个决策树桩
    :return:
    """
    rightCnt = 0
    for data in dataSet:
        pre = adaPredic(data, G)
        if pre == data[-1]:
            rightCnt += 1
    return rightCnt / float(len(dataSet))


def main():
    dataSet = getDataSet()
    for t in [3, 5, 11]:   # 学习器的数量
        G = AdaBoost(dataSet, t)
        print(f"G{t} = {G}")
        print(calcAcc(dataSet, G))


if __name__ == '__main__':
    main()
