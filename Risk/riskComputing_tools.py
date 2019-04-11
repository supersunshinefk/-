import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import json
import threading
import config
import pandas as pd
import os
import sys
import random

import config

sched_delay = config.sched_delay
# 预测学习算法，可用于风险及风险影响角预测函数的训练
# timemat_：历史数据的时间列表矩阵
# risklist_：历史数据的风险值
# return：训练的模型参数
def functiontrain(timemat_, risklist_):
    # 将历史数据分为训练集和验证集
    param_train, param_test, risk_train, risk_test = train_test_split(timemat_, risklist_, random_state=1)

    # 模型训练
    riskpredict = LinearRegression()
    riskpredict.fit(param_train, risk_train)

    # 模型误差
    risk_pred = riskpredict.predict(param_test)
    mse = metrics.mean_squared_error(risk_test, risk_pred)

    # 返回模型参数和误差
    return [[riskpredict.intercept_, riskpredict.coef_], mse]


# 对于历史数据采用不同的模型进行拟合，选择拟合误差最小的回归模型作为最终的预测函数
# historymat：各历史时刻及其相对应的预测值矩阵吗（m,n），m:样本个数，n:特征个数+1（目前项目始终为2）
# return：最优的模型的参数列表
def predictmodel(timehistory, historypreval):
    funcresult = []

    # y=a0+a1*t
    funcresult.append(
        functiontrain(timehistory.reshape(len(timehistory), 1), historypreval.reshape(len(timehistory), 1)))

    # y=a0+a1*t+a2*t^2
    timehistory_ = np.c_[timehistory, np.square(timehistory)]
    funcresult.append(functiontrain(timehistory_, historypreval))

    # y=a0+a1*t+a2*t^2+a3*t^3
    timehistory_ = np.c_[timehistory_, np.power(timehistory, 3)]
    funcresult.append(functiontrain(timehistory_, historypreval))

    # 获取训练误差最小的模型参数并返回
    funcresult = np.array(funcresult)

    modelparam = funcresult[funcresult[:, 1] == funcresult[:, 1].min(), 0]

    return modelparam


# 根据模型，重组待预测数据并返回预测结果（目前可用于风险预测，各参数趋势）
# modelparamlist：训练后的模型的参数列表
# pretime：要预测的时刻
# return：预测值
def preresult(modelparamlist, pretime):
    paramval = modelparamlist[0][1]

    if (len(paramval) == 3):
        pretime = [pretime, pretime ** 2, pretime ** 3]
    elif (len(paramval) == 2):
        pretime = [pretime, pretime ** 2]

    preval = np.sum(np.array(pretime) * np.array(paramval)) + modelparamlist[0][0]

    return preval



# 计算均方根，用于求上下风险边界和当前风险值及波动值
# paramlist：各参数某一时刻的取值（或各参数经处理后的综合值）
# return：所有参数的均方根
def rms(paramlist):
    return round(np.sqrt(np.mean(np.square(paramlist))), 1)


# 计算当风险触及风险边界时，风险发生的概率
# upperbound：风险上界--既定值，不许预测
# lowerbound：风险下界--既定值，不许预测
# currentriskval_1：预测时刻的风险值--通过预测风险值与时间的关系，得出预测时刻的风险值
# currentriskval_0：预测时刻前一时刻的风险值
# fluctval：波动范围--既定值，不许预测
# timeCell_mean：平均时间间隔
# return:预测时刻的风险概率值
def current_risk_value(param_list, upper_bound, min_max):

    #min_max = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [410, 410, 1000, 1000, 1000, 1500, 1500, 1500, 1510, 1500]]

    # 计算当前基础风险值
    param_list = list(param_list)

    riskValue = np.sqrt(np.mean(np.square(param_list)))

    #计算当前风险值距离风险上界的距离
    if(riskValue >= upper_bound):

        return riskValue
    else:
        riskValue_distance_upper = upper_bound - riskValue

        #生成记录各参数距离其上界的距离百分比列表
        distance_list = []
        for i in range(len(param_list)):
            if(min_max[1][i] - min_max[0][i] == 0):

                distance_list.append(0)

            else:
                if(min_max[1][i] > param_list[i]):

                    result = (min_max[1][i] - param_list[i])/(min_max[1][i] - min_max[0][i])
                    if(result >= 0.5):
                        distance_list.append(0)
                    elif(0.4 <= result < 0.5):
                        distance_list.append(0.1)
                    elif (0.3 <= result < 0.4):
                        distance_list.append(0.2)
                    elif (0.2 <= result < 0.3):
                        distance_list.append(0.4)
                    elif (0.1 <= result < 0.2):
                        distance_list.append(0.6)
                    elif (0 <= result < 0.1):
                        distance_list.append(0.8)
                else:
                    result = (param_list[i] - min_max[1][i]) / (min_max[1][i] - min_max[0][i])
                    if (result >= 0.5):
                        distance_list.append(1.55)
                    elif (0.4 <= result < 0.5):
                        distance_list.append(1.45)
                    elif (0.3 <= result < 0.4):
                        distance_list.append(1.35)
                    elif (0.2 <= result < 0.3):
                        distance_list.append(1.25)
                    elif (0.1 <= result < 0.2):
                        distance_list.append(1.15)
                    elif (0 <= result < 0.1):
                        distance_list.append(1.05)

        distance_upper = np.max(distance_list)
        current_risk_value = riskValue + distance_upper * riskValue_distance_upper

    return round(current_risk_value, 1)



def riskProbability(upperbound, lowerbound, currentriskval_1, currentriskval_0, fluctval, timeCell_mean):

    impangle = np.abs(impactangle(currentriskval_1, currentriskval_0, timeCell_mean))

    if(currentriskval_1 <= lowerbound):
        riskPro = 0.1 + random.uniform(0, 0.02)

    elif(upperbound <= currentriskval_1):
        riskPro = 0.92 + random.uniform(0, 0.02)

    else:

        if(upperbound - currentriskval_1 >= fluctval * np.sin(impangle)):

            riskPro = (np.exp((currentriskval_1 - lowerbound) / (upperbound - lowerbound))-1) / 2


        else:
            proA = (fluctval * np.sin(impangle) - (upperbound - currentriskval_1)) / (fluctval * np.sin(impangle))

            proB = (4 * fluctval) / ((upperbound - lowerbound) * np.pi)

            riskPro = (np.exp((currentriskval_1 - lowerbound) / (upperbound - lowerbound))-1) / 2 + proA * proB


    if (riskPro >= 1):
        riskPro = 0.92 + random.uniform(0., 0.02)
    if type(riskPro) is np.ndarray:
        return round(riskPro[0], 2)
    else:
        return round(riskPro, 2)

# 计算各参数的波动矩阵
# historymatrix：各参数的200个（暂定）历史样本矩阵，历史矩阵（m,n）m为参数个数，n为样本个数
# return：各参数样本数据中相邻时刻的波动值


def fluctmat(historymatrix):
    fluctmatrix = []
    for i in range(historymatrix.shape[0]):
        paramfluct = []
        for j in range(historymatrix.shape[1] - 1):
            paramfluct.append(np.abs(historymatrix[i, j + 1] - historymatrix[i, j]))
        fluctmatrix.append(paramfluct)
    return np.mat(fluctmatrix)


# 计算各参数波动的最大值
# historymatrix：各参数的200个（暂定）历史样本矩阵
# return：各参数风险波动的最大值列表
def fluctextremevaluemat(historymatrix):
    fluctmatrix = fluctmat(historymatrix)
    extremevaluematrix = []
    for i in range(fluctmatrix.shape[0]):
        extremevaluematrix.append(fluctmatrix[i].max())

    return extremevaluematrix


# 影响角计算
# currentriskval_1:当前时刻风险值
# currentriskval_0:前一时刻风险值
# timeCell:时间间隔
# return：影响角
def impactangle(currentriskval_1, currentriskval_0, timeCell):

    if (timeCell == 0):
        impangle = 0
    else:
        impangle = 180 * np.arctan((currentriskval_1 - currentriskval_0) / timeCell) / np.pi
    return np.around(impangle, 2)


# data:历史数据
# min_max：系统数据
# return：风险计算的各个参数值
def computing(cal_data, min_max):

    #min_max = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1100, 120, 190, 800, 1.6, 1.6, 200000, 10000, 2000, 2000, 20000, 20000]]

    cal_data = cal_data.transpose()
    data = cal_data.iloc[:, 1:cal_data.shape[1]]
    # 计算风险上下边界

    upperbound = rms(min_max[1])
    lowerbound = rms(min_max[0])

    normalizing = 100 / (upperbound - lowerbound)
    # 计算各历史时刻的风险值
    historyRisk = []
    for i in range(data.shape[0]):
        currVal = current_risk_value(data.iloc[i], upperbound, min_max)
        historyRisk.append(normalizing * (currVal - lowerbound))
    historyRisk = np.array(historyRisk)
    #print('historyRisk', historyRisk)

    currentRisk = historyRisk[-1]
    lastRisk = historyRisk[-2]

    # 计算影响角
    historyTime_1 = cal_data.iat[cal_data.shape[0] - 1, 0]
    historyTime_0 = cal_data.iat[cal_data.shape[0] - 2, 0]
    timeCell = (historyTime_1 - historyTime_0) * sched_delay

    angle = impactangle(currentRisk, lastRisk, timeCell)

    if (angle == 0):
        angle = 3

    # 计算风险波动上下界
    fluctVal = rms(fluctextremevaluemat(np.array(data.transpose()))) + (upperbound - lowerbound) / 5
    fluctVal = normalizing * fluctVal

    # 拟合风险预测曲线，计算各预测时刻的风险预测值，指定给出51个预测数据，用于计算50个风险概率值
    timeList = sched_delay * np.arange(data.shape[0])
    param = predictmodel(timeList, historyRisk)

    preTimeList = np.arange(51)
    riskValList = []
    for i in preTimeList:
        time = sched_delay * preTimeList[i]
        riskVal = preresult(param, time)
        riskValList.append(riskVal)
    #print('riskValList',riskValList)
    # 预测各时刻风险概率值
    timeCell_mean = sched_delay
    riskProList = []
    for i in range(50):

        riskProVal = riskProbability(100, 0, riskValList[i + 1],
                                     riskValList[i], fluctVal, timeCell_mean)
        riskProList.append(riskProVal)

    # 将结果映射到[0, 100]

    if (currentRisk < 0):
        currentRisk = 0

    radAngle = np.pi * (angle) / 180
    riskFluctUp = round(currentRisk + np.abs(fluctVal * np.sin(radAngle)), 2)
    riskFluctLow = round(currentRisk - np.abs(fluctVal * np.sin(radAngle)), 2)

    if (currentRisk > 150):
        fluctVal = 5
        currentRisk = 145
        riskFluctUp = round(currentRisk + np.abs(fluctVal * np.sin(radAngle)), 2)
        riskFluctLow = round(currentRisk - np.abs(fluctVal * np.sin(radAngle)), 2)

    if (riskFluctUp > 150):
        riskFluctUp = 150
        riskFluctLow = round(currentRisk - (150 - currentRisk), 2)
        fluctVal = round((150 - currentRisk), 2)

    if (riskFluctLow < 0):
        riskFluctLow = np.abs(riskFluctLow)

    # 坐标
    midpointCoordinates = [0, round(currentRisk, 2)]

    if (radAngle < 0):
        upperCoordinates = [round(-fluctVal * np.cos(radAngle), 2), round(- fluctVal * np.sin(radAngle), 2)]
        lowerCoordinates = [round(fluctVal * np.cos(radAngle), 2), round(fluctVal * np.sin(radAngle), 2)]
    else:
        upperCoordinates = [round(fluctVal * np.cos(radAngle), 2), round(fluctVal * np.sin(radAngle), 2)]
        lowerCoordinates = [round(-fluctVal * np.cos(radAngle), 2), round(-fluctVal * np.sin(radAngle), 2)]

    result = [round(currentRisk, 2), riskProList, lowerbound, upperbound, angle, riskFluctUp, riskFluctLow, midpointCoordinates,
              upperCoordinates, lowerCoordinates]

    return result
