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
# Author : Xunjunhe & LiKang
# Description : 稳定性、稳定性梯度、稳定性累积效应、关联图谱


import numpy as np
import time
from numpy import linalg as LA
import redis
import json
import pandas as pd
import re
from threading import Thread
import config
import sys
from daemon import Daemon
from ast import literal_eval

program_type = config.ProgramEnum.stability.name


class StabilityDaemon(Daemon):

    def __init__(self, project=None):
        Daemon.__init__(self, project, program_type)
        self.project_id = None
        self.object_list = None
        self.data_organizer = None
        self.para_reference_name_list = None
        self.script = None
        self.new_parameter_name_list = None
        self.para_list = None
        self.min_max = None
    def set_project_info(self, project_info):

        try:
            #self.script = project_info["data_organizer"]["script"]
            self.project_id = project_info["project_id"]
            # 获取object组
            self.object_list = project_info["object"]
            self.para_list = project_info["analysis_parameter"]["script"]
            # 获取组织器脚本
            self.data_organizer = project_info["data_organizer"]


            self.script = project_info["data_organizer"]["script"]["analysis_parameter"]
            min_ = []  # 最小值
            max_ = []  # 最大值
            self.min_max = []  # 最小值和最大值列表
            for list_ in self.para_list:
                min_.append(list_['para_quota_min'])
                max_.append(list_['para_quota_max'])

            self.min_max.append(min_)
            self.min_max.append(max_)
        except Exception as e:
            print("Failed to get the project info：%s" % e)

    def Multi_Linear_Regress(self,All_data):
        graph = []
        for i in range(len(All_data)):
            lines = []
            for j in range(len(All_data)):

                y = np.array(All_data[i]).T
                B = np.array(All_data[j]).T
                if np.sum(B)==0:
                    B+=0.0001

                B1 = np.hstack((np.ones((len(B), 1)), B))
                # print(np.dot(B1.T, B1))
                B2 = np.hstack((B1, B ** 2))
                B3 = np.hstack((B2, B ** 3))

                y_hat1 = np.dot(B1, np.dot(np.dot(np.linalg.inv(np.dot(B1.T, B1)), B1.T), y))
                y_hat2 = np.dot(B2, np.dot(np.dot(np.linalg.inv(np.dot(B2.T, B2)), B2.T), y))
                y_hat3 = np.dot(B3, np.dot(np.dot(np.linalg.inv(np.dot(B3.T, B3)), B3.T), y))

                r_square1 = 1 - np.sqrt(np.sum((y - y_hat1) ** 2) / np.sum(y ** 2))
                r_square2 = 1 - np.sqrt(np.sum((y - y_hat2) ** 2) / np.sum(y ** 2))
                r_square3 = 1 - np.sqrt(np.sum((y - y_hat3) ** 2) / np.sum(y ** 2))

                Rs = np.array([r_square1, r_square2, r_square3])
                Rs=np.nan_to_num(Rs)
                if len(Rs[Rs > 0.8]) == 0:
                    Value = np.max(Rs)
                    position = Rs.tolist().index(Value)
                else:
                    Value = Rs[Rs > 0.8][0]
                    position = Rs.tolist().index(Value)
                lines.append([Value, position])
            graph.append(lines)
        return graph

    def stab_correlation(self, value_matrix):
        columns_value = value_matrix.shape[1]  # 列数
        rows_value = value_matrix.shape[0] - 1  # 参数个数
        var_value = np.zeros(rows_value)
        var_value0 = np.zeros(rows_value)
        var_t3 = np.zeros(rows_value)  # t3时段各参数方差
        var_t2 = np.zeros(rows_value)
        var_t1 = np.zeros(rows_value)
        var_t0 = np.zeros(rows_value)
        var_speed3 = np.zeros(rows_value)  # t3时段各参数方差梯度
        var_speed2 = np.zeros(rows_value)
        var_speed1 = np.zeros(rows_value)
        var_cumulation = np.zeros(rows_value)  # t3时刻各参数方差累积效应
        var_cumulation0 = np.zeros(rows_value)

        # graph=self.Multi_Linear_Regress(value_matrix[1:])

        # 矩阵预处理
        for i in range(rows_value):
            if (np.max(value_matrix[i + 1]) == 0):
                value_matrix[i + 1] = np.zeros(columns_value)
            else:
            #     value_matrix[i + 1] = (value_matrix[i + 1] - np.min(value_matrix[i + 1])) / (
            #             np.max(value_matrix[i + 1]) - np.min(value_matrix[i + 1]))
                value_matrix[i + 1] = value_matrix[i + 1] / np.max(value_matrix[i + 1])

        for i in range(rows_value):

            var_value[i] = np.cov((value_matrix[i + 1, 2:columns_value]))
            var_value0[i] = np.cov(value_matrix[i + 1, 1:columns_value - 1])
            var_t3[i] = np.round(np.cov(value_matrix[i + 1, 3:columns_value]), 3)
            var_t2[i] = np.round(np.cov(value_matrix[i + 1, 2:columns_value - 1]), 3)
            var_t1[i] = np.round(np.cov(value_matrix[i + 1, 1:columns_value - 2]), 3)
            var_t0[i] = np.round(np.cov(value_matrix[i + 1, 0:columns_value - 3]), 3)

            if (value_matrix[i + 1, rows_value - 1] - value_matrix[i + 1, rows_value - 2]) == 0:
                var_speed3[i] = 0
            else:
                var_speed3[i] = (var_t3[i] - var_t2[i]) / (
                        value_matrix[i + 1, rows_value - 1] - value_matrix[i + 1, rows_value - 2])
            var_speed3 = np.round(var_speed3, 2)
            if (value_matrix[i + 1, rows_value - 2] - value_matrix[i + 1, rows_value - 3]) == 0:
                var_speed2[i] = 0
            else:
                var_speed2[i] = (var_t2[i] - var_t1[i]) / (
                        value_matrix[i + 1, rows_value - 2] - value_matrix[i + 1, rows_value - 3])
            if (value_matrix[i + 1, rows_value - 3] - value_matrix[i + 1, rows_value - 4]) == 0:
                var_speed1[i] = 0
            else:
                var_speed1[i] = (var_t1[i] - var_t0[i]) / (
                        value_matrix[i + 1, rows_value - 3] - value_matrix[i + 1, rows_value - 4])
            if (var_speed3[i] - var_speed2[i]) == 0:
                var_cumulation[i] = 0
            else:
                var_cumulation[i] = (var_t3[i] - var_t2[i]) / (var_speed3[i] - var_speed2[i])

            var_cumulation = np.round(var_cumulation, 2)
            if (var_speed2[i] - var_speed1[i]) == 0:
                var_cumulation0[i] = 0
            else:
                var_cumulation0[i] = (var_t2[i] - var_t1[i]) / (var_speed2[i] - var_speed1[i])
            var_cumulation0 = np.round(var_cumulation0, 2)

        # var_matrix = np.zeros(64).reshape(8,8)
        var_matrix = np.cov(value_matrix[1:columns_value, 0:rows_value])
        correlation_coefficient_matrix = np.corrcoef(value_matrix[1:columns_value + 1, ])
        correlation_coefficient_matrix = np.nan_to_num(correlation_coefficient_matrix)
        correlation_coefficient_matrix = correlation_coefficient_matrix.tolist()
        # print(var_matrix)
        Eigenvalues, Eigenvectors = LA.eig(var_matrix)
        Eigenvalues = Eigenvalues.real.tolist()
        Max_pos = Eigenvalues.index(max(Eigenvalues))
        Max_vector = Eigenvectors[:, Max_pos]
        Max_vector = Max_vector ** 2
        # Rt_vector = (0.2*(var_value ** 2) + 1.8*(var_speed3 ** 2) + 1.8*(var_cumulation ** 2) + 0.2*(Max_vector ** 2))
        # Rt_vector0 = (var_value0 ** 2 + var_speed2 ** 2 + var_cumulation0 ** 2 + Max_vector ** 2)  # Max_vector t-1与t一样
        # Rt = sum(Rt_vector) / rows_value * 2.5
        # Rt0 = (sum(Rt_vector0)) / rows_value * 2.5
        # Rt_speed = (Rt - Rt0) / (value_matrix[0, rows_value - 1] - value_matrix[0, rows_value - 2])
        # var_speed3 = np.round(100 + (var_speed3 - 1000 * (max_list - min_list)) /
        #                       (1000 * (max_list - min_list)) * 38.2, 2)
        # var_cumulation = np.round(100 + (
        #         (var_cumulation - 1000000 * (max_list - min_list)) / (1000000 *
        #                                                               (max_list - min_list))) * 38.2, 2)
        # print("What_is_this?>>>>>>>",sum(var_value ** 2),sum(var_speed3 ** 2),sum(var_cumulation ** 2),sum(Max_vector ** 2))

        Rt_vector = (10 * (var_value ** 2) + 100 * (var_speed3 ** 2) + 100 * (var_cumulation ** 2) + 0.1 * (
                Max_vector ** 2))
        Rt_vector0 = (10 * (var_value0 ** 2) + 100 * (var_speed2 ** 2) + 100 * (var_cumulation0 ** 2) + 0.1 * (
                Max_vector ** 2))  # Max_vector t-1与t一样
        Rt = sum(Rt_vector) / rows_value * 2.5
        Rt0 = (sum(Rt_vector0)) / rows_value * 2.5
        Rt_speed = (Rt - Rt0) / (value_matrix[0, rows_value - 1] - value_matrix[0, rows_value - 2])
        # var_speed3_real = var_speed3
        # var_cumulation_real = var_cumulation
        var_speed3 = np.round(100 + (var_speed3 - 3) / 3 * 38.2, 2)
        var_cumulation = np.round(100 + ((var_cumulation - 3) / 3) * 38.2, 2)
        Rt = 200 / (1 + 2 ** (Rt * 4))
        var_cumulation = var_cumulation.tolist()
        # var_cumulation = var_cumulation[0]
        if (type(var_cumulation[0]) == list):
            var_cumulation = var_cumulation[0]

        for elecun in var_cumulation:

            if elecun > 100:
                var_cumulation[var_cumulation.index(elecun)] = 100
            if elecun < 0:
                var_cumulation[var_cumulation.index(elecun)] = 0
        # print(var_value, var_speed3, var_cumulation, correlation_coefficient_matrix, Rt, Rt_speed)
        var_speed3 = var_speed3.tolist()
        # var_speed3 = var_speed3[0]
        if (type(var_speed3[0]) == list):
            var_speed3 = var_speed3[0]
        for elecun1 in var_speed3:
            if elecun1 > 100:
                var_speed3[var_speed3.index(elecun1)] = 100
            if elecun1 < 0:
                var_speed3[var_speed3.index(elecun1)] = 0

        # Rt---> Indx
        # Rt_speed ---> Grad
        # var_value---->Vari_Elem
        # var_speed3---->Vari_Grad_Elem
        # correlation_coefficient_matrix------>Vari_Matr
        # var_cumulation->Vari_Cumn_Elem
        stability_dict = dict()
        stability_dict['Indx'] = round(Rt, 2).real
        stability_dict['Grad'] = round(Rt_speed, 2).real
        # Stab['Vari_Elem'] = var_value.tolist()
        stability_dict['Vari_Grad_Elem'] = var_speed3
        stability_dict['Vari_Matr'] = np.round(correlation_coefficient_matrix, 2).tolist()
        stability_dict['Vari_Cumn_Elem'] = var_cumulation
        # stability_dict['Vari_Grad_Elem_real'] = var_speed3_real
        # stability_dict['Vari_Cumn_Elem_real'] = var_cumulation_real
        # print(Stab)
        
        return stability_dict

    def process_stability(self, sub_organizer_channel):

        ps = config.redis_conn.pubsub()
        ps.subscribe(sub_organizer_channel)

        for item in ps.listen():

            sub_type = item["type"]
            if sub_type == "subscribe":
                continue

            data_block_id = item["data"]
            if isinstance(data_block_id, bytes):
                data_block_id = data_block_id.decode("utf-8")
            else:
                data_block_id = str(data_block_id)

            channel = item["channel"].decode("utf-8")  # b'channel:dispenser:PRJ002:862631039192314'
            channel_list = channel.split(":")
            project_id = channel_list[2]
            object_id = channel_list[3]

            organizer_queue_name = config.Hset_Organizer_Analysis_QueueName.format(project_id=self.project_id, object_id=object_id)
            # if config.redis_conn.hlen(organizer_queue_name) >= config.Max_Dispenser_Queue_Size:
            print(">>>start time", time.time())

            while True:

                key_list = config.redis_conn.hkeys(organizer_queue_name)
                if key_list:
                    break

            max_key = max(key_list)
            while True:
                data = config.redis_conn.hget(organizer_queue_name, max_key)
                if data:
                    break
                else:
                    print("The stability get the data to be empty!")
                    time.sleep(5)
            data = json.loads(data.decode('utf-8'))
            value_matrix = np.mat(data)


            stability_dict = self.stab_correlation(value_matrix)
            if stability_dict and isinstance(stability_dict, dict):
                stability_json = json.dumps(stability_dict)
                stability_queue_name = config.HSet_StabQueueName.format(project_id=project_id,
                                                                        object_id=object_id)
            print('stability_json is ---->', stability_json)
            config.redis_conn.hset(stability_queue_name, str(data_block_id), stability_json)

            # 判断redis队列中长度是否超过Max_Compute_Queue_Size条，超过以后,进行删除最旧的一条计算数据
            if config.redis_conn.hlen(stability_queue_name) > config.Max_Compute_Queue_Size:
                key_list = config.redis_conn.hkeys(stability_queue_name)
                new_key_list = []
                for keys in key_list:

                    key_min = keys.decode('utf-8')
                    new_key_list.append(int(key_min))
                key = min(new_key_list)
                config.redis_conn.hdel(stability_queue_name, key)
            else:
                pass
            # 存储数据至INFLUXDB数据库中
            json_body = [
                        {
                            "measurement": "Stability",
                            "tags": {
                                "object": "%s:%s" %(project_id,object_id),
                            },
                            # "time": "2017-03-12T22:00:00Z",
                            "fields": {
                                "savetime":str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                                "stability":str(stability_dict)
                            }
                        }
            ]
            config.influxdb_conn.write_points(json_body) 

            print("end time", time.time())

    def run(self):
        #parameter_name_list = re.findall(config.regex, self.script)

        new_parameter_name_list = []
        for parameter_name in self.script:
            para = parameter_name[0]
            new_parameter_name_list.append(para)
        sub_organizer_channel = []

        if self.object_list is not None:
            for object_dict in self.object_list:
                client_id = str(object_dict["object_id"])
                # thread_list.append(Thread(target=self.send_data, args=(new_parameter_name_list, client_id)))
                channel_name = config.Channel_OrganizerName.format(project_id=self.project_id,
                                                                   object_id=client_id)
                sub_organizer_channel.append(channel_name)

            self.process_stability(sub_organizer_channel)


def main(project_queue_id):
    """
    :param project_queue_id: redis中保存的project队列id
    :return:
    """
    # 获取消息队列数据
    res = config.redis_conn.hget(config.HSET_BOSS_PROJECT, project_queue_id)
    if not res:
        print('Get Project Info Error!')
        raise SystemExit(1)
    else:
        # str(d, encoding='utf-8')
        res = res.decode("utf-8")
        project_info = json.loads(res)
        if project_info:
            project_status = project_info["server"]
            project_id = project_info["project_id"]
            print("  >>{0},project_id={1},status={2}".format(program_type, project_id, project_status))
            stab_daemon = StabilityDaemon(project=project_id)
            stab_daemon.set_project_info(project_info)
            stab_daemon.start()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} [start|stop]'.format(sys.argv[0]), file=sys.stderr)
        raise SystemExit(1)

    queue_id = sys.argv[1]
    main(queue_id)
