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
# Author :XiaoJunhe & LiKang
# Description : 趋势算法

import numpy as np
import re
from sklearn import linear_model
import json
import threading
import sys
import config
from daemon import Daemon
import time
from ast import literal_eval

from util import linear_regression_util as lru
import pandas as pd

program_type = config.ProgramEnum.tendency.name


class TendencyDaemon(Daemon):
    def __init__(self, project):

        Daemon.__init__(self, project, program_type)
        self.object_list = None
        self.project_id = None
        self.object_list = None
        self.script = None
        self.new_parameter_name_list = None

    # 获取boss数据
    def set_project_info(self, project_info):
        try:
            self.project_id = project_info["project_id"]
            # 获取object组
            self.object_list = project_info["object"]
            self.para_list = project_info["analysis_parameter"]["script"]
            # 获取组织器脚本
            self.data_organizer = project_info["data_organizer"]

            self.parameter_list = project_info["analysis_parameter"]["script"]

            self.script = project_info["data_organizer"]["script"]["analysis_parameter"]
            min_ = []  # 最小值
            max_ = []  # 最大值
            self.min_max = []  # 最小值和最大值列表
            for list_ in self.para_list:
                min_.append(list_['para_quota_min'])
                max_.append(list_['para_quota_max'])

            self.min_max.append(min_)
            self.min_max.append(max_)
        except Exception as err:
            print("Failed to get the project info!", err)


    def sigmoid(self, x):

        return 1.0 / (1.0 + np.exp(-x))

    # 定义发送数据的任务
    def lstm_predict(self, value_matrix):
        #tendency_element_list = []
        tendency_prediction_list = []
        tendency_dict = {}

        #new_matrix = np.hstack((value_matrix, value_matrix, value_matrix, value_matrix, value_matrix, value_matrix, value_matrix))
        dataframe = pd.DataFrame(value_matrix[1:, :]).transpose()
        #dataframe.to_csv('G:/new_matrix.csv', index = False)
        result = []
        nb_epoch = 15#迭代次数
        seq_len = 40#步长
        batchSize = 100
        lstmFirstLayer = 32#第一层lstm每步神经元个数
        lstmSecondLayer = 32#第二层lstm每步神经元个数
        future_predict_num = 40#连续预测未来30个数据
        dataset = dataframe.values

        for i in range(dataset.shape[1]):
            #predictions, testScore = ttu.lstm_run(nb_epoch, seq_len, batchSize, lstmFirstLayer, lstmSecondLayer, future_predict_num, [dataset[:, i]])

            predictions, testScore = lru.line_regression_run(seq_len, future_predict_num, np.array([dataset[:, i]]).reshape(-1, len([dataset[:, i]])))
            predictions = predictions.reshape(-1, len(predictions)) #线性回归专用
            predictions = np.round(predictions[0, :], 2).tolist()

            result.append(predictions)
            prop_basic = 100 * (1.5 - self.sigmoid(testScore))
            prop = [prop_basic * 0.99, prop_basic * 0.92, prop_basic * 0.85]

            #print(i, prop)
            tendency_array = np.array(
                [[min(predictions[0:10]), (sum(predictions[0:10]) / 10), max(predictions[0: 10]), prop[0]],
                 [min(predictions[10: 20]), (sum(predictions[10:20]) / 10), max(predictions[10: 20]),
                  prop[1]],
                 [min(predictions[20: 30]), (sum(predictions[20:30]) / 10), max(predictions[20: 30]),
                  prop[2]]])

            tendency_array = np.round(tendency_array, 2).tolist()

            for i in range(3):
                tendency_array[i][3] = "%.2f%%" % (tendency_array[i][3])
            tendency_list = list(tendency_array)
            element_tendency_dict = dict()
            element_tendency_dict["m10"] = tendency_list[0]
            element_tendency_dict["m30"] = tendency_list[1]
            element_tendency_dict["m60"] = tendency_list[2]
            #element_tendency_dict["m120"] = tendency_array[3]
            tendency_prediction_list.append(element_tendency_dict)

            # 把Tend_Elem_list添加到字典中

            tendency_dict["Tend_Elem"] = result
            tendency_dict["Tend_Pred"] = tendency_prediction_list



        return tendency_dict

    def process_tendency(self, sub_organizer_channel):
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

            organizer_queue_name = config.Hset_Organizer_Analysis_QueueName.format(
                project_id=project_id, object_id=object_id)

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
                    print("The tendency get the data to be empty!")
                    time.sleep(5)
            data = json.loads(data.decode('utf-8'))
            value_matrix = np.mat(data)

            tendency_dict = self.lstm_predict(value_matrix)
            print(tendency_dict)
            tendency_json = json.dumps(tendency_dict)

            tendency_queue_name = config.HSet_TendQueueName.format(project_id=project_id,
                                                                   object_id=object_id)
            print("tendency_queue_name:", tendency_queue_name, data_block_id)
            config.redis_conn.hset(tendency_queue_name, str(data_block_id), tendency_json)

            # 判断redis队列中长度是否超过Max_Compute_Queue_Size条，超过以后,进行删除最旧的一条计算数据
            if config.redis_conn.hlen(tendency_queue_name) > config.Max_Compute_Queue_Size:
                key_list = config.redis_conn.hkeys(tendency_queue_name)
                new_key_list = []
                for keys in key_list:
                    key_min = keys.decode('utf-8')
                    new_key_list.append(int(key_min))
                key = min(new_key_list)
                config.redis_conn.hdel(tendency_queue_name, key)
            else:
                pass

                # 存储数据至INFLUXDB数据库中
            json_body = [
                {
                    "measurement": "Tendency",
                    "tags": {
                        "object": "%s:%s" % (project_id, object_id),
                    },
                    # "time": "2017-03-12T22:00:00Z",
                    "fields": {
                        "savetime":str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                        "tendency":str(tendency_dict)
                    }
                }
            ]
            config.influxdb_conn.write_points(json_body)

            print("end time", time.time())

    def run(self):
        new_parameter_name_list = []
        for parameter_name in self.script:
            para = parameter_name[0]
            new_parameter_name_list.append(para)
        self.new_parameter_name_list = new_parameter_name_list

        sub_organizer_channel = []
        if self.object_list is not None:
            for object_dict in self.object_list:
                client_id = str(object_dict["object_id"])
                channel_name = config.Channel_OrganizerName.format(project_id=self.project_id,
                                                                   object_id=client_id)

                sub_organizer_channel.append(channel_name)

            self.process_tendency(sub_organizer_channel)


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

            tend_daemon = TendencyDaemon(project=project_id)
            tend_daemon.set_project_info(project_info)
            tend_daemon.start()


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: {} [start|stop]'.format(sys.argv[0]), file=sys.stderr)
        raise SystemExit(1)
    queue_id = sys.argv[1]
    main(queue_id)
