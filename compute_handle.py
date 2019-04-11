###############################################################################
# Copyright (c) 2014-2018 Mixlinker Networks Inc. <mixiot@mixlinker.com>
# All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Application License of Mixlinker Networks License and Mixlinker
# Distribution License which accompany this distribution.
#
# The Mixlinker License is available at
#    http://www.mixlinker.com/legal/license.html
# and the Mixlinker Distribution License is available at
#    http://www.mixlinker.com/legal/distribution.html
#
# Contributors:
#    Mixlinker Technical Team
###############################################################################
# Date : 2018-5-29
# Author : HuWenping
# Description : organizer计算脚本


#encoding = utf-8
import numpy as np
import pandas as pd
import config
#1：加法
#2：减法
#3：乘法
#4：除法
#5：积分
#实现计算，返回处理后的数据集
#new_origin_data：加入优化目标后的历史数据集
#origin_data：原始数据
#symbol:计算标识
#parameter_index_list:参与运算的参数索引列表
#ssymbol_id:函数被首次调用标识
#return:处理后的数据集
"""
                value_matrix = np.mat(param_list)
                value_matrix = pd.DataFrame(value_matrix)
                new_value_matrix = pd.read_csv('G:/1-new_value_matrix.csv')#获取
                new_value_matrix = dh.organize_dataset(new_value_matrix, value_matrix, 5, [4,3],1)
                value_matrix = np.mat(new_value_matrix)

"""
sched_delay = config.sched_delay ## 获取数据时间


def organize_dataset_first(origin_data, symbol, parameter_index_list):
    origin_data = pd.DataFrame(origin_data)
    time_num = 30

    goal_param = []
    if (symbol == 'addition'):

        for i in range(origin_data.shape[1]):
            result = origin_data.iat[parameter_index_list[0], i]
            for j in range(len(parameter_index_list) - 1):
                result += origin_data.iat[parameter_index_list[j + 1], i]
            goal_param.append(round(result, 2))

    elif (symbol == 'subtraction'):

        for i in range(origin_data.shape[1]):
            result = origin_data.iat[parameter_index_list[0], i]
            for j in range(len(parameter_index_list) - 1):
                result -= origin_data.iat[parameter_index_list[j + 1], i]
            goal_param.append(round(result, 2))

    elif (symbol == 'multiplication'):

        for i in range(origin_data.shape[1]):
            result = origin_data.iat[parameter_index_list[0], i]
            for j in range(len(parameter_index_list) - 1):
                result *= origin_data.iat[parameter_index_list[j + 1], i]
            goal_param.append(result)

    elif (symbol == 'ratio'):

        for i in range(origin_data.shape[1]):
            if (origin_data.iat[parameter_index_list[1], i] == 0):
                result = round(origin_data.iat[parameter_index_list[0], i] / (origin_data.iat[parameter_index_list[1], i] + 1), 2)
            else:
                result = round(origin_data.iat[parameter_index_list[0], i] / origin_data.iat[parameter_index_list[1], i], 2)
            goal_param.append(round(result, 2))

    elif (symbol == 'integration'):

        for i in range(origin_data.shape[1]):
            parameter_one = 0
            parameter_two = 0
            if(i < time_num):
                if(i == 0):
                    parameter_one = origin_data.iat[parameter_index_list[0], i]
                    parameter_two = origin_data.iat[parameter_index_list[1], i]

                else:
                    for j in range(i):
                        time_interval = (origin_data.iat[0, j + 1] - origin_data.iat[0, j]) * sched_delay
                        parameter_one += (origin_data.iat[parameter_index_list[0], j + 1] + origin_data.iat[parameter_index_list[0], j]) * time_interval * 0.5
                        parameter_two += (origin_data.iat[parameter_index_list[1], j + 1] + origin_data.iat[parameter_index_list[1], j]) * time_interval * 0.5

            else:
                for j in range(time_num):
                    time_interval = (origin_data.iat[0, i - j] - origin_data.iat[0, i - j - 1]) * sched_delay
                    parameter_one += (origin_data.iat[parameter_index_list[0], i - j] + origin_data.iat[parameter_index_list[0], i - j - 1]) * time_interval * 0.5
                    parameter_two += (origin_data.iat[parameter_index_list[1], i - j] + origin_data.iat[parameter_index_list[1], i - j - 1]) * time_interval * 0.5

            if (parameter_two == 0):
                each_value = round((parameter_one / (parameter_two + 1)), 2)
            else:
                each_value = round(parameter_one / parameter_two, 2)
            goal_param.append(each_value)
    elif (symbol == 'integration_up'):

        for i in range(origin_data.shape[1]):
            parameter_one = 0
            parameter_two = 0
            if(i < time_num):
                if(i == 0):
                    parameter_one = origin_data.iat[parameter_index_list[0], i]
                    parameter_two = origin_data.iat[parameter_index_list[1], i]

                else:
                    for j in range(i):
                        time_interval = (origin_data.iat[0, j + 1] - origin_data.iat[0, j]) * sched_delay
                        parameter_one += (origin_data.iat[parameter_index_list[0], j + 1] + origin_data.iat[parameter_index_list[0], j]) * time_interval * 0.5
                    parameter_two += (origin_data.iat[parameter_index_list[1], i] - origin_data.iat[parameter_index_list[1], 0])

            else:
                for j in range(time_num):
                    time_interval = (origin_data.iat[0, i - j] - origin_data.iat[0, i - j - 1]) * sched_delay
                    parameter_one += (origin_data.iat[parameter_index_list[0], i - j] + origin_data.iat[parameter_index_list[0], i - j - 1]) * time_interval * 0.5
                parameter_two += (origin_data.iat[parameter_index_list[1], i] - origin_data.iat[parameter_index_list[1], i - time_num])

            if (parameter_two == 0):
                each_value = round((parameter_one / (parameter_two + 1)), 2)
            else:
                each_value = round(parameter_one / parameter_two, 2)
            goal_param.append(each_value)

    elif (symbol == 'integration_low'):
         for i in range(origin_data.shape[1]):
            parameter_one = 0
            parameter_two = 0
            if(i < time_num):
                if(i == 0):
                    parameter_one = origin_data.iat[parameter_index_list[0], i]
                    parameter_two = origin_data.iat[parameter_index_list[1], i]

                else:
                    for j in range(i):
                        time_interval = (origin_data.iat[0, j + 1] - origin_data.iat[0, j]) * sched_delay

                        parameter_two += (origin_data.iat[parameter_index_list[1], j + 1] + origin_data.iat[parameter_index_list[1], j]) * time_interval * 0.5
                    parameter_one += (origin_data.iat[parameter_index_list[0], i] - origin_data.iat[parameter_index_list[0], 0])
            else:
                for j in range(time_num):
                    time_interval = (origin_data.iat[0, i - j] - origin_data.iat[0, i - j - 1]) * sched_delay

                    parameter_two += (origin_data.iat[parameter_index_list[1], i - j] + origin_data.iat[parameter_index_list[1], i - j - 1]) * time_interval * 0.5
                parameter_one += (origin_data.iat[parameter_index_list[0], i] - origin_data.iat[parameter_index_list[0], i - time_num])

            if (parameter_two == 0):
                each_value = round((parameter_one / (parameter_two + 1)), 2)
            else:
                each_value = round(parameter_one / parameter_two, 2)
            goal_param.append(each_value)

    elif (symbol == 'integration_non'):

        for i in range(origin_data.shape[1]):
            parameter_one = 0
            parameter_two = 0
            if(i < time_num):
                if(i == 0):
                    parameter_one = origin_data.iat[parameter_index_list[0], i]
                    parameter_two = origin_data.iat[parameter_index_list[1], i]

                else:
                    parameter_one = origin_data.iat[parameter_index_list[0], i] - origin_data.iat[parameter_index_list[0], 0]
                    parameter_two = origin_data.iat[parameter_index_list[1], i] - origin_data.iat[parameter_index_list[1], 0]
            else:
                parameter_one = origin_data.iat[parameter_index_list[0], i] - origin_data.iat[parameter_index_list[0], i - time_num]
                parameter_two = origin_data.iat[parameter_index_list[1], i] - origin_data.iat[parameter_index_list[1], i - time_num]

            if (parameter_two == 0):
                each_value = round((parameter_one / (parameter_two + 1)), 2)
            else:
                each_value = round(parameter_one / parameter_two, 2)
            goal_param.append(each_value)

    # new_origin_data = np.r_[np.array(origin_data), np.array(goal_param).reshape(1, len(goal_param))]
    return goal_param



# def organize_dataset(origin_data, symbol, parameter_index_list, lastData):
#
#     time_num = 10
#     #lastData = origin_data.iloc[1:, origin_data.shape[1] - 1].tolist()
#     if (symbol == 'addition'):
#
#         result = lastData[parameter_index_list[0] - 1]
#
#         for j in range(len(parameter_index_list) - 1):
#             result += lastData[parameter_index_list[j + 1] - 1]
#         lastData.append(round(result, 2))
#
#     elif (symbol == 'subtraction'):
#
#         result = lastData[parameter_index_list[0] - 1]
#
#         for j in range(len(parameter_index_list) - 1):
#             result -= lastData[parameter_index_list[j + 1] - 1]
#
#         lastData.append(round(result, 2))
#
#     elif (symbol == 'multiplication'):
#
#         result = lastData[parameter_index_list[0] - 1]
#         for j in range(len(parameter_index_list) - 1):
#             result *= lastData[parameter_index_list[j + 1] - 1]
#
#         lastData.append(round(result, 2))
#
#     elif (symbol == 'ratio'):
#
#         if(lastData[parameter_index_list[1]] == 0):
#             result = round(lastData[parameter_index_list[0]] / lastData[parameter_index_list[1]] + 1, 2)
#         else:
#             result = round(lastData[parameter_index_list[0]] / lastData[parameter_index_list[1]], 2)
#         lastData.append(result)
#
#     elif (symbol == 'integration'):
#
#         parameter_one = 0
#         parameter_two = 0
#         for i in range(time_num):
#
#             time_interval = (origin_data.iat[0, origin_data.shape[1] - 1 - i] - origin_data.iat[0, origin_data.shape[1] - 2 - i]) * 15
#             parameter_one += ((origin_data.iat[parameter_index_list[0], origin_data.shape[1] - 1 - i] + origin_data.iat[parameter_index_list[0], origin_data.shape[1] - 2 - i]) * time_interval * 0.5)
#             parameter_two += ((origin_data.iat[parameter_index_list[1], origin_data.shape[1] - 1 - i] + origin_data.iat[parameter_index_list[1], origin_data.shape[1] - 2 - i]) * time_interval * 0.5)
#
#         if (parameter_two == 0):
#             each_value = round((parameter_one / (parameter_two + 1)), 2)
#         else:
#             each_value = round(parameter_one / parameter_two, 2)
#         lastData.append(each_value)
#
#     else:
#         return lastData
#
#     return lastData

def organize_dataset(origin_data, symbol, parameter_index_list):
    origin_data = pd.DataFrame(origin_data)
    result = 0
    time_num = 30
    #lastData = origin_data.iloc[1:, origin_data.shape[1] - 1].tolist()
    if (symbol == 'addition'):

        result = origin_data.iat[parameter_index_list[0], -1]

        for j in range(len(parameter_index_list) - 1):
            result += origin_data.iat[parameter_index_list[j + 1], -1]
        #lastData.append(round(result, 2))

    elif (symbol == 'subtraction'):

        result = origin_data.iat[parameter_index_list[0], -1]

        for j in range(len(parameter_index_list) - 1):
            result -= origin_data.iat[parameter_index_list[j + 1], -1]

        #lastData.append(round(result, 2))

    elif (symbol == 'multiplication'):

        result = origin_data.iat[parameter_index_list[0], -1]
        for j in range(len(parameter_index_list) - 1):
            result *= origin_data.iat[parameter_index_list[j + 1], -1]

        #lastData.append(round(result, 2))

    elif (symbol == 'ratio'):

        if(origin_data.iat[parameter_index_list[1], -1] == 0):
            result = round(origin_data[parameter_index_list[0], -1] / origin_data[parameter_index_list[1], -1] + 1, 2)
        else:
            result = round(origin_data[parameter_index_list[0], -1] / origin_data[parameter_index_list[1], -1], 2)
        #lastData.append(result)

    elif (symbol == 'integration'):

        parameter_one = 0
        parameter_two = 0
        for i in range(time_num):

            time_interval = (origin_data.iat[0, origin_data.shape[1] - 1 - i] - origin_data.iat[0, origin_data.shape[1] - 2 - i]) * sched_delay
            parameter_one += ((origin_data.iat[parameter_index_list[0], origin_data.shape[1] - 1 - i] + origin_data.iat[parameter_index_list[0], origin_data.shape[1] - 2 - i]) * time_interval * 0.5)
            parameter_two += ((origin_data.iat[parameter_index_list[1], origin_data.shape[1] - 1 - i] + origin_data.iat[parameter_index_list[1], origin_data.shape[1] - 2 - i]) * time_interval * 0.5)

        if (parameter_two == 0):
            result = round((parameter_one / (parameter_two + 1)), 2)
        else:
            result = round(parameter_one / parameter_two, 2)
        #lastData.append(each_value)


    elif (symbol == 'integration_up'):
        parameter_one = 0
        parameter_two = 0
        for i in range(time_num):

            time_interval = (origin_data.iat[0, origin_data.shape[1] - 1 - i] - origin_data.iat[0, origin_data.shape[1] - 2 - i]) * sched_delay
            parameter_one += ((origin_data.iat[parameter_index_list[0], origin_data.shape[1] - 1 - i] + origin_data.iat[parameter_index_list[0], origin_data.shape[1] - 2 - i]) * time_interval * 0.5)
        parameter_two += (origin_data.iat[parameter_index_list[1], origin_data.shape[1] - 1] - origin_data.iat[parameter_index_list[1], origin_data.shape[1] - time_num - 1])

        if (parameter_two == 0):
            result = round((parameter_one / (parameter_two + 1)), 2)
        else:
            result = round(parameter_one / parameter_two, 2)
        #lastData.append(each_value)

    elif (symbol == 'integration_low'):
        parameter_one = 0
        parameter_two = 0
        for i in range(time_num):

            time_interval = (origin_data.iat[0, origin_data.shape[1] - 1 - i] - origin_data.iat[0, origin_data.shape[1] - 2 - i]) * sched_delay

            parameter_two += ((origin_data.iat[parameter_index_list[1], origin_data.shape[1] - 1 - i] + origin_data.iat[parameter_index_list[1], origin_data.shape[1] - 2 - i]) * time_interval * 0.5)
        parameter_one += (origin_data.iat[parameter_index_list[0], origin_data.shape[1] - 1] - origin_data.iat[parameter_index_list[0], origin_data.shape[1] - time_num - 1])

        if (parameter_two == 0):
            result = round((parameter_one / (parameter_two + 1)), 2)
        else:
            result = round(parameter_one / parameter_two, 2)
        #lastData.append(each_value)

    elif (symbol == 'integration_non'):
        parameter_one = 0
        parameter_two = 0

        parameter_one = origin_data.iat[parameter_index_list[0], origin_data.shape[1] - 1] - origin_data.iat[parameter_index_list[0], origin_data.shape[1] - time_num - 1]
        parameter_two = origin_data.iat[parameter_index_list[1], origin_data.shape[1] - 1] - origin_data.iat[parameter_index_list[1], origin_data.shape[1] - time_num - 1]

        if (parameter_two == 0):
            result = round((parameter_one / (parameter_two + 1)), 2)
        else:
            result = round(parameter_one / parameter_two, 2)
        #lastData.append(each_value)

    return result

# 程序首次执行，调用该方法
# 返回300列的矩阵
# origin_data：原始数据
# symbol_list：运算符号列表
# computing_list参与运算的参数矩阵
# def final_organize_dataset_first(origin_data, symbol_list, computing_list):
#     for i in range(len(computing_list)):
#         origin_data = pd.DataFrame(organize_dataset_first(origin_data, symbol_list[i], computing_list[i]))
#     return origin_data
#
# #程序首次执行之后调用该方法
# #return：更新后的数据集
# #origin_data：原始数据
# #symbol_list：运算符号列表
# #computing_list参与运算的参数矩阵
# def final_organize_dataset(origin_data, symbol_list, computing_list, new_origin_data):
#
#     lastData = origin_data.iloc[1:, origin_data.shape[1] - 1].tolist()
#     for i in range(len(computing_list)):
#         lastData = organize_dataset(origin_data, symbol_list[i], computing_list[i], lastData)
#
#     new_origin_data = new_origin_data.iloc[1:, 1:]
#     new_origin_data = np.c_[np.array(new_origin_data), np.array(lastData)]
#     new_origin_data = np.r_[np.array(origin_data.iloc[0, :]).reshape(1, origin_data.shape[1]), new_origin_data]
#
#     return new_origin_data
