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
# Date : 2018-8-15
# Author :XiaoJunhe
# Description : index, grad, cumulation, partail grad, partail cumulation
# 设备状态映像图,指数累积效应曲线,指数累积效应曲线,指数累积效应曲线,运行指数曲线指数梯度曲线

import json
import re
import threading
import config
import redis
import sys
import time
from ast import literal_eval
from daemon import Daemon
import numpy as np
import time

program_type = config.ProgramEnum.index.name


class IndexDaemon(Daemon):

    def __init__(self, project):
        Daemon.__init__(self, project, program_type)
        self.object_list = None
        self.parameter_list = None
        self.project_id = None
        self.script = None

    def set_project_info(self, data):
        self.object_list = data["object"]
        self.project_id = data["project_id"]

        self.parameter_list = data["analysis_parameter"]

        self.script = data["data_organizer"]["script"]["analysis_parameter"]
        min_ = []  # 最小值
        max_ = []  # 最大值
        self.min_max = []  # 最小值和最大值列表
        for list_ in self.parameter_list:
            min_.append(list_['para_quota_min'])
            max_.append(list_['para_quota_max'])

        self.min_max.append(min_)
        self.min_max.append(max_)

    def index_correlation(self, value_matrix, AHP_weight):

        # 客观赋权法——标准离差法
        origin_values = np.array(value_matrix[1:])
        trans_values = []
        for k in range(len(origin_values)):
            max_abs_value = np.max(np.abs(origin_values[k]))
            min_abs_value = np.min(np.abs(origin_values[k]))
            if max_abs_value == min_abs_value:
                nond_value = origin_values[k]
            else:
                nond_value = (origin_values[k] - min_abs_value) / (max_abs_value - min_abs_value)
            trans_values.append(nond_value)
        trans_values = np.array(trans_values)
        sigma = np.diag(np.cov(trans_values))
        normalized_weight = []
        # 主观赋权法——AHP层次分析法
        AHP_weight = np.array(AHP_weight)
        # 综合赋权法
        weights = np.sqrt(AHP_weight ** 2 + sigma ** 2).tolist()
        for m in range(len(weights)):
            result = weights[m] / (sum(weights))
            # 权重值归一化，weight为参数权重值数组，normalied_weight为归一化后的参数权重值数组
            normalized_weight.append(result)

        # 列数
        columns = value_matrix.shape[1]
        # 参数个数
        rows = value_matrix.shape[0] - 1
        index3 = 0
        index2 = 0
        index1 = 0
        timedelta1_2 = (value_matrix[0, columns - 1] - value_matrix[0, columns - 2])  # timedelta1_2为t3时刻与t2时刻的时间差
        timedelta2_3 = (value_matrix[0, columns - 2] - value_matrix[0, columns - 3])
        for i in range(rows):
            index3 += value_matrix[i + 1, columns - 3] * normalized_weight[i]
            index2 += value_matrix[i + 1, columns - 2] * normalized_weight[i]
            index1 += value_matrix[i + 1, columns - 1] * normalized_weight[i]

        # grad1为当前时刻指数的梯度, grad2为前一时刻指数的梯度
        grad1 = (index1 - index2) / timedelta1_2
        grad2 = (index2 - index3) / timedelta2_3
        # cumulation为当前时刻指数的累积效应
        cumulation = (grad1 - grad2) / timedelta1_2

        partail_grad1 = np.zeros(rows)
        partail_grad2 = np.zeros(rows)
        partail_cumulation = np.zeros(rows)
        for j in range(rows):
            # partail_grad方向梯度
            if (value_matrix[j + 1, columns - 1] - value_matrix[j + 1, columns - 2]) == 0:
                partail_grad1[j] = 0
            else:
                partail_grad1[j] = (index1 - index2) / (
                        value_matrix[j + 1, columns - 1] - value_matrix[j + 1, columns - 2])

            if (value_matrix[j + 1, columns - 2] - value_matrix[j + 1, columns - 3]) == 0:
                partail_grad2[j] = 0
            else:
                partail_grad2[j] = (index2 - index3) / (
                        value_matrix[j + 1, columns - 2] - value_matrix[j + 1, columns - 3])

            # partail_cumulation方向累积效应
            if (value_matrix[j + 1, columns - 1] - value_matrix[j + 1, columns - 2]) == 0:
                partail_cumulation[j] = 0
            else:
                partail_cumulation[j] = (partail_grad1[j] - partail_grad2[j]) / (
                        value_matrix[j + 1, columns - 1] - value_matrix[j + 1, columns - 2])

        partail_grad1 = 100 + ((partail_grad1 - 3) / 3) * 38.2
        for l in range(rows):
            if partail_grad1[l] < 0:
                partail_grad1[l] = 0
            if partail_grad1[l] > 100:
                partail_grad1[l] = 100

        partail_cumulation = 100 + ((partail_cumulation - 3) / 3) * 38.2
        for l in range(rows):
            if partail_cumulation[l] < 0:
                partail_cumulation[l] = 0
            if partail_cumulation[l] > 100:
                partail_cumulation[l] = 100

        index_dict = dict()
        grad = round(grad1, 2).real
        if np.isnan(grad):
            grad = 0.0
        if np.isinf(grad):
            grad = 0.0
        index_dict["Grad"] = grad
        index_dict["Indx"] = round(index1, 2).real
        index_dict["Grad_Elem"] = np.round(partail_grad1, 2).tolist()

        cumu = round(cumulation, 2).real
        if np.isnan(cumu):
            cumu = 0.0
        if np.isinf(cumu):
            cumu = 0.0
        index_dict["Cumu"] = cumu
        index_dict["Cumu_Elem"] = np.round(partail_cumulation, 2).tolist()

        return index_dict

    def process_index(self, sub_organizer_channel, weight_list, new_parameter_name_list):
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

            organizer_queue_name = config.Hset_Organizer_Analysis_QueueName.format(project_id=project_id,
                                                                                   object_id=object_id)
            # if config.redis_conn.hlen(organizer_queue_name) >= config.Max_Dispenser_Queue_Size:

            print(">>>开始时间", time.time())
            while True:

                key_list = config.redis_conn.hkeys(organizer_queue_name)
                if key_list:
                    break
            # key_list = list(map(int, key_list))
            max_key = max(key_list)
            while True:
                data = config.redis_conn.hget(organizer_queue_name, max_key)
                if data:
                    break
                else:
                    print("map 获取数据为空！！！")
                    time.sleep(5)
            data = json.loads(data.decode('utf-8'))
            value_matrix = np.mat(data)

            index_dict = self.index_correlation(value_matrix, weight_list)

            index_json = json.dumps(index_dict)
            index_queue_name = config.HSet_IndexQueueName.format(project_id=project_id,
                                                                 object_id=object_id)
            print("index_queue_name:", index_queue_name, data_block_id)
            config.redis_conn.hset(index_queue_name, str(data_block_id), index_json)

            # 判断redis队列中长度是否超过Max_Compute_Queue_Size条，超过以后,进行删除最旧的一条计算数据
            if config.redis_conn.hlen(index_queue_name) > config.Max_Compute_Queue_Size:

                key_list = config.redis_conn.hkeys(index_queue_name)
                key = min(key_list)
                config.redis_conn.hdel(index_queue_name, key)
            else:
                pass
            print(">>>结束时间", time.time())

    def run(self):
        # 获取对象列表以及项目id
        object_list = self.object_list
        parameter_list = self.parameter_list

        # 获取权重数据列表
        weight_list = []
        for parameter in parameter_list:
            weight = parameter["para_weight"]
            weight_list.append(weight)

        # parameter_name_list = re.findall(config.regex, self.script)
        new_parameter_name_list = []
        for parameter_name in self.script:
            para = parameter_name[0]
            new_parameter_name_list.append(para)

        sub_organizer_channel = []
        for object_dict in object_list:
            client_id = str(object_dict["object_id"])

            channel_name = config.Channel_OrganizerName.format(project_id=self.project_id,
                                                               object_id=client_id)
            sub_organizer_channel.append(channel_name)

        self.process_index(sub_organizer_channel, weight_list, new_parameter_name_list)


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
            index_daemon = IndexDaemon(project=project_id)
            index_daemon.set_project_info(data=project_info)
            index_daemon.start()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} [start|stop]'.format(sys.argv[0]), file=sys.stderr)
        raise SystemExit(1)
    queue_id = sys.argv[1]
    main(queue_id)
