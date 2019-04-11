# coding: utf-8

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
# Description : risk 算法
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics """
import json
import threading
import config
import sys
import pandas as pd
import numpy as np
from ast import literal_eval
program_type = config.ProgramEnum.risk.name
from Risk import riskComputing_tools
from daemon import Daemon
import time
import re



class RiskDaemon(Daemon):
    def __init__(self, project=None):
        Daemon.__init__(self, project, program_type)
        self.project_id = None
        self.object_list = None
        self.para_reference_name_list = None
        self.new_parameter_name_list = None
        self.script = None
        self.min_max = None

    def set_project_info(self, project_info):
        try:
            # 获取project_id
            self.project_id = project_info["project_id"]
            # 获取object组
            self.object_list = project_info["object"]
            self.para_list = project_info["analysis_parameter"]
            # 获取组织器脚本
            self.data_organizer = project_info["data_organizer"]

            self.parameter_list = project_info["analysis_parameter"]["script"]

            self.script = project_info["data_organizer"]["script"]["analysis_parameter"]
            min_ = []  # 最小值
            max_ = []  # 最大值
            self.min_max = []  # 最小值和最大值列表
            for list_ in self.parameter_list:
                
                min_.append(list_['para_quota_min'])
                max_.append(list_['para_quota_max'])

            self.min_max.append(min_)
            self.min_max.append(max_)
        except Exception as e:
            print("Failed to get the project info：%s" % e)

    def process_risk(self, sub_organizer_channel):
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

            print(">>>start time", time.time())
            organizer_queue_name = config.Hset_Organizer_Analysis_QueueName.format(project_id=project_id, object_id=object_id)

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
                    print("The risk get the data to be empty!")
                    time.sleep(5)
            data = json.loads(data.decode('utf-8'))
            # data = config.redis_conn.hgetall(organizer_queue_name)  # 转换成dict格式

            risk_result = riskComputing_tools.computing(pd.DataFrame(data), self.min_max)
            if risk_result:
                risk_dict = dict()
                risk_dict['Risk'] = risk_result[0]
                risk_dict['Risk_Fore'] = risk_result[1]
                risk_dict['Risk_Diag'] = risk_result[2:7]

                risk_dict['Risk_Midpoint'] = risk_result[7]  # 中间坐标
                risk_dict['Risk_Upper'] = risk_result[8]  # 上坐标
                risk_dict['Risk_Lower'] = risk_result[-1:][0]  # 下坐标
                risk_result_json = json.dumps(risk_dict)
                risk_queue_name = config.HSet_RiskQueueName.format(project_id=project_id,
                                                                   object_id=object_id)
                config.redis_conn.hset(risk_queue_name, data_block_id, risk_result_json)

                # 判断redis队列中长度是否超过Max_Compute_Queue_Size条，超过以后,进行删除最旧的一条计算数据
                if config.redis_conn.hlen(risk_queue_name) > config.Max_Compute_Queue_Size:
                    key_list = config.redis_conn.hkeys(risk_queue_name)
                    new_key_list = []
                    for keys in key_list:

                        key_min = keys.decode('utf-8')
                        new_key_list.append(int(key_min))
                    key = min(new_key_list)
                    config.redis_conn.hdel(risk_queue_name, key)
                else:
                    pass
                
                json_body = [
                    {
                        "measurement": "Risk",
                        "tags": {
                            "object": "%s:%s" %(project_id,object_id),
                        },
                        # "time": "2017-03-12T22:00:00Z",
                        "fields": {
                            "savetime":str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                            "risk":str(risk_dict)
                        }
                    }
                ]
                config.influxdb_conn.write_points(json_body)  # 写入数据，同时创建表

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

            self.process_risk(sub_organizer_channel)


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

            risk_daemon = RiskDaemon(project=project_id)
            risk_daemon.set_project_info(project_info)
            risk_daemon.start()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} [start|stop]'.format(sys.argv[0]), file=sys.stderr)
        raise SystemExit(1)
    queue_id = sys.argv[1]
    main(queue_id)
