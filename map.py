##############################################################################
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
# Description : map图显示，雷达图，指数


import json
import re
import redis
import config
import numpy as np
import pandas as pd
import sys
import time
from daemon import Daemon
from ast import literal_eval

program_type = config.ProgramEnum.map.name


class MapDaemon(Daemon):
    def __init__(self, project=None):
        Daemon.__init__(self, project, program_type)
        self.object_list = None
        self.project_id = None
        self.object_list = None
        self.script = None
        self.new_parameter_name_list = None

        self.mean_list = None
        self.min_list = None
        self.max_list = None
        self.mode_list = None

    def set_project_info(self, project_info):
        try:

            self.object_list = project_info["object"]
            self.project_id = project_info["project_id"]

            self.parameter_list = project_info["analysis_parameter"]

            self.script = project_info["data_organizer"]["script"]["analysis_parameter"]
        except Exception as e:
            print("Failed to get project info!：%s" % e)

    def map_mean(self, value_matrix):

        # 参数个数
        rows_value = value_matrix.shape[0] - 1

        mean_list = []
        min_list = []
        max_list = []  # 最大值列表
        mode_list = []
        for i in range(rows_value):
            origin_mean = np.mean(value_matrix[i + 1])
            origin_min = np.min(value_matrix[i + 1])
            origin_max = np.max(value_matrix[i + 1])
            enlarge_min = 2 * origin_min - origin_mean
            enlarge_max = 2 * origin_max - origin_mean
            mean_list.append(origin_mean)
            min_list.append(enlarge_min)
            max_list.append(enlarge_max)

        pd_matrix = pd.DataFrame(value_matrix)

        for i in range(1, rows_value + 1):
            mode_list.append((pd_matrix.loc[i].mode()).iloc[-1])

        self.mean_list = mean_list
        self.min_list = min_list
        self.max_list = max_list
        self.mode_list = mode_list

    def map_correlation(self, value_matrix):
        # 列数
        columns_value = value_matrix.shape[1]

        # 参数个数
        rows_value = value_matrix.shape[0] - 1

        mean_list = []
        min_list = []
        max_list = []  # 最大值列表
        mode_list = []
        origin_min_list = []
        origin_max_list = []
        for i in range(rows_value):
            origin_mean = np.mean(value_matrix[i + 1])
            origin_min = np.min(value_matrix[i + 1])
            origin_max = np.max(value_matrix[i + 1])
            enlarge_min = 4 * origin_min - 3 * origin_mean
            enlarge_max = 4 * origin_max - 3 * origin_mean
            origin_min_list.append(origin_min)
            origin_max_list.append(origin_max)
            mean_list.append(origin_mean)
            min_list.append(enlarge_min)
            max_list.append(enlarge_max)

        pd_matrix = pd.DataFrame(value_matrix)
        for i in range(1, rows_value + 1):
            mode_list.append((pd_matrix.loc[i].mode()).iloc[-1])

        # 需要展示的参数原始值
        value_list = value_matrix[1:rows_value + 1, columns_value - 1]
        value_list = value_list.tolist()

        my_list = []

        for i in range(rows_value):
            if max_list[i] - min_list[i] == 0:
                if mean_list[i] == 0:
                    value_list_i = value_list[i]
                    result = (value_list_i[0] + 1) / (mean_list[i] + 1) * 61.8
                else:
                    value_list_i = value_list[i]
                    result = value_list_i[0] / mean_list[i] * 61.8

                my_list.append(result)
            else:
                value_list_i = value_list[i]
                value_std = (value_list_i[0] - min_list[i]) / (max_list[i] - min_list[i])
                Mean_std = (mean_list[i] - min_list[i]) / (max_list[i] - min_list[i])
                if Mean_std > 0.618:
                    result = 61.8 / Mean_std * value_std
                else:
                    result = 100 * (0.382 * value_std + 0.618 - Mean_std) / (1 - Mean_std)
                # result = 100 * ((((Mean_std - 0.618) / (Mean_std - Mean_std ** 2)) * value_std + (
                #    (0.618 - Mean_std ** 2) / (Mean_std - Mean_std ** 2))) * value_std)
                my_list.append(result)

            if my_list[i] < 0:
                my_list[i] = 0
            if my_list[i] > 100:
                my_list[i] = 100
        return my_list, value_list

    def process_map(self, sub_organizer_channel):
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
                    print("The Map get the data to be empty!！")
                    time.sleep(5)

            data = json.loads(data.decode('utf-8'))
            value_matrix = np.mat(data)

            # 程序只跑一趟
            # self.map_mean(value_matrix)

            map_list, value_list = self.map_correlation(value_matrix)

            map_list = np.round(np.mat(map_list), 2).tolist()[0]
            value_list = np.round(np.mat(value_list).T, 2).tolist()[0]

            map_dict = {}
            map_dict["map_transform_data"] = map_list
            map_dict["map_original_data"] = value_list
            print(map_list, value_list)

            map_json = json.dumps(map_dict)

            map_queue_name = config.HSet_MapQueueName.format(project_id=project_id,
                                                             object_id=object_id)
            config.redis_conn.hset(map_queue_name, str(data_block_id), map_json)
                # 判断redis队列中长度是否超过Max_Compute_Queue_Size条，超过以后,进行删除最旧的一条计算数据
            if config.redis_conn.hlen(map_queue_name) > config.Max_Compute_Queue_Size:
                key_list = config.redis_conn.hkeys(map_queue_name)
                new_key_list = []
                for keys in key_list:

                    key_min = keys.decode('utf-8')
                    new_key_list.append(int(key_min))
                key = min(new_key_list)
                config.redis_conn.hdel(map_queue_name, key)
            else:
                pass

            json_body = [
                {
                    "measurement": "Map",
                    "tags": {
                        "object": "%s:%s" %(project_id,object_id),
                    },
                    # "time": "2017-03-12T22:00:00Z",
                    "fields": {
                        "savetime":str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                        "map":str(map_dict)
                    }
                }
            ]
            config.influxdb_conn.write_points(json_body)
            print(">>>end time", time.time())

    def run(self):
        # parameter_name_list = re.findall(config.regex, self.script)
        new_parameter_name_list = []
        # for parameter_name in self.parameter_list:
        #     para = parameter_name[1:]
        #     new_parameter_name_list.append(para)
        # self.new_parameter_name_list = new_parameter_name_list

        new_parameter_name_list = []
        for parameter_name in self.script:
            para = parameter_name[0]
            new_parameter_name_list.append(para)
        
        sub_organizer_channel = []
        if self.object_list is not None:
            for object_dict in self.object_list:
                client_id = str(object_dict["object_id"])
                channel_name = config.Channel_OrganizerName.format(project_id=self.project_id,
                                                                   object_id=client_id)
                sub_organizer_channel.append(channel_name)

            self.process_map(sub_organizer_channel)


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
            map_daemon = MapDaemon(project=project_id)
            map_daemon.set_project_info(project_info)
            map_daemon.start()


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: {} [start|stop]'.format(sys.argv[0]), file=sys.stderr)
        raise SystemExit(1)
    queue_id = sys.argv[1]
    main(queue_id)
