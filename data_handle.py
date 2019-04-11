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
# Author :HuWenping
# Description : organizer计算脚本

import numpy as np


# 1：加法
# 2：减法
# 3：乘法
# 4：除法
# 5：积分
# 实现计算，返回处理后的数据集
# new_origin_data：加入优化目标后的历史数据集
# origin_data：原始数据
# symbol:计算标识
# parameter_index_list:参与运算的参数索引列表
# ssymbol_id:函数被首次调用标识
# return:处理后的数据集
"""
                value_matrix = np.mat(param_list)
                value_matrix = pd.DataFrame(value_matrix)
                new_value_matrix = pd.read_csv('G:/1-new_value_matrix.csv')#获取
                new_value_matrix = dh.organize_dataset(new_value_matrix, value_matrix, 5, [4,3],1)
                value_matrix = np.mat(new_value_matrix)

"""


def organize_dataset(new_origin_data, origin_data, symbol, parameter_index_list, symbol_id):
    time_num = 10

    if (symbol_id == 1):

        goal_param = []
        if (symbol == 1):

            for i in range(origin_data.shape[1]):
                result = origin_data.iat[parameter_index_list[0], i]
                for j in range(len(parameter_index_list) - 1):
                    result += origin_data.iat[parameter_index_list[j + 1], i]
                goal_param.append(round(result, 2))

        elif (symbol == 2):

            for i in range(origin_data.shape[1]):
                result = origin_data.iat[parameter_index_list[0], i]
                for j in range(len(parameter_index_list) - 1):
                    result -= origin_data.iat[parameter_index_list[j + 1], i]
                goal_param.append(round(result, 2))

        elif (symbol == 3):

            for i in range(origin_data.shape[1]):
                result = origin_data.iat[parameter_index_list[0], i]
                for j in range(len(parameter_index_list) - 1):
                    result *= origin_data.iat[parameter_index_list[j + 1], i]
                goal_param.append(result)

        elif (symbol == 4):

            for i in range(origin_data.shape[1]):
                if (origin_data.iat[parameter_index_list[1], i] == 0):
                    result = round(
                        origin_data.iat[parameter_index_list[0], i] / (origin_data.iat[parameter_index_list[1], i] + 1),
                        2)
                else:
                    result = round(
                        origin_data.iat[parameter_index_list[0], i] / origin_data.iat[parameter_index_list[1], i], 2)
                goal_param.append(round(result, 2))

        elif (symbol == 5):

            for i in range(origin_data.shape[1]):
                parameter_one = 0
                parameter_two = 0
                if (i < time_num):
                    if (i == 0):
                        parameter_one = origin_data.iat[parameter_index_list[0], i]
                        parameter_two = origin_data.iat[parameter_index_list[1], i]

                    else:
                        for j in range(i):
                            time_interval = (origin_data.iat[0, j + 1] - origin_data.iat[0, j]) * 15
                            parameter_one += (origin_data.iat[parameter_index_list[0], j + 1] + origin_data.iat[
                                parameter_index_list[0], j]) * time_interval * 0.5
                            parameter_two += (origin_data.iat[parameter_index_list[1], j + 1] + origin_data.iat[
                                parameter_index_list[1], j]) * time_interval * 0.5

                else:
                    for j in range(time_num):
                        time_interval = (origin_data.iat[0, i - j] - origin_data.iat[0, i - j - 1]) * 15
                        parameter_one += (origin_data.iat[parameter_index_list[0], i - j] + origin_data.iat[
                            parameter_index_list[0], i - j - 1]) * time_interval * 0.5
                        parameter_two += (origin_data.iat[parameter_index_list[1], i - j] + origin_data.iat[
                            parameter_index_list[1], i - j - 1]) * time_interval * 0.5

                if (parameter_two == 0):
                    each_value = round((parameter_one / (parameter_two + 1)), 2)
                else:
                    each_value = round(parameter_one / parameter_two, 2)
                goal_param.append(each_value)

        else:
            return origin_data

        new_origin_data = np.r_[np.array(origin_data), np.array(goal_param).reshape(1, len(goal_param))]
        return new_origin_data

    else:
        lastData = origin_data.iloc[1:, origin_data.shape[1] - 1].tolist()
        if (symbol == 1):

            result = lastData[parameter_index_list[0] - 1]
            for j in range(len(parameter_index_list) - 1):
                result += lastData[parameter_index_list[j + 1] - 1]
            lastData.append(round(result, 2))

        elif (symbol == 2):

            result = lastData[parameter_index_list[0] - 1]

            for j in range(len(parameter_index_list) - 1):
                result -= lastData[parameter_index_list[j + 1] - 1]

            lastData.append(round(result, 2))

        elif (symbol == 3):

            result = lastData[parameter_index_list[0] - 1]
            for j in range(len(parameter_index_list) - 1):
                result *= lastData[parameter_index_list[j + 1] - 1]

            lastData.append(round(result, 2))

        elif (symbol == 4):

            if (lastData[parameter_index_list[1]] == 0):
                result = round(lastData[parameter_index_list[0]] * lastData[parameter_index_list[1]] + 1, 2)
            else:
                result = round(lastData[parameter_index_list[0]] * lastData[parameter_index_list[1]], 2)
            lastData.append(result)

        elif (symbol == 5):

            parameter_one = 0
            parameter_two = 0
            for i in range(time_num):
                time_interval = (origin_data.iat[0, origin_data.shape[1] - 1 - i] - origin_data.iat[
                    0, origin_data.shape[1] - 2 - i]) * 15
                parameter_one += ((origin_data.iat[parameter_index_list[0], origin_data.shape[1] - 1 - i] +
                                   origin_data.iat[
                                       parameter_index_list[0], origin_data.shape[1] - 2 - i]) * time_interval * 0.5)
                parameter_two += ((origin_data.iat[parameter_index_list[1], origin_data.shape[1] - 1 - i] +
                                   origin_data.iat[
                                       parameter_index_list[1], origin_data.shape[1] - 2 - i]) * time_interval * 0.5)

            if (parameter_two == 0):
                each_value = round((parameter_one / (parameter_two + 1)), 2)
            else:
                each_value = round(parameter_one / parameter_two, 2)
            lastData.append(each_value)

        else:
            return origin_data

        new_origin_data = new_origin_data.iloc[1:, 1:]
        new_origin_data = np.c_[np.array(new_origin_data), np.array(lastData)]
        new_origin_data = np.r_[
            np.array(origin_data.iloc[0, :]).reshape(1, len(origin_data.iloc[0, :])), new_origin_data]
        return new_origin_data
