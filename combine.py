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
# Author : Roy & XinJinquan
# Description : combine 数据重组


import json
import re
import redis
import threading
import config
import sys
import time
from ast import literal_eval
from daemon import Daemon

program_type = config.ProgramEnum.combine.name


class CombineDaemon(Daemon):
    def __init__(self, project=None):
        Daemon.__init__(self, project, program_type)
        self.object_list = None
        self.project_id = None
        self.parameter_list = None
        self.script = None

        self.parm_list = None
        self.opt_parm_list = []
        self.parm_en_list = None
        self.analysis_parm_cn_list = None
        self.analysis_parm_cn_list = None
        self.opt_parm_en_list = None
        self.analysis_parm_en_list = None
        self.account_login = None

    def set_project_info(self, project_info):
        try:
            # 获取对象列表以及项目id
            self.object_list = project_info["object"]
            self.project_id = project_info["project_id"]
            self.analysis_parameter_list = project_info["analysis_parameter"]["script"]
            self.opt_parameter_list = project_info["optimization_parameter"]["script"]
            self.analysis_script = project_info["data_organizer"]["script"]["analysis_parameter"]
            self.optimization_script = project_info["data_organizer"]["script"]["optimization_parameter"]
            self.account_login = project_info['account']['account_login']
        except Exception as e:
            print("Failed to get the project info：%s" % e)

    # 定义数据发送任务
    def combine_data_block(self, project_id, object_id, data_block_id):

        combine_dict = dict()
        try:
            map_id_list = config.redis_conn.hkeys(config.HSet_MapQueueName.format(project_id=project_id, 
                                                              object_id=object_id))
            if map_id_list:
                map_id = int(max(map_id_list))
                map_data = config.redis_conn.hget(
                    config.HSet_MapQueueName.format(project_id=project_id, object_id=object_id),
                    map_id)
                if map_data is None:
                    map_data = []
                elif type(map_data) is bytes:
                    map_data = map_data.decode('utf-8')
                    map_data = json.loads(map_data)
                combine_dict["Map"] = map_data
        except Exception as error:
            print('The Map data is empty!', error)

        # 指数数据
        try:
            index_id_list = config.redis_conn.hkeys(config.HSet_IndexQueueName.format(project_id=project_id, 
                                                              object_id=object_id))
            if index_id_list:
                index_id = int(max(index_id_list))
                index_data = config.redis_conn.hget(config.HSet_IndexQueueName.format(project_id=project_id,
                                                                                      object_id=object_id),
                                                    index_id)
                if index_data is None:
                    index_data = []
                elif isinstance(index_data, bytes):
                    index_data = index_data.decode('utf-8')
                    index_data = json.loads(index_data)
                combine_dict["Indx"] = index_data
        except Exception as error:
            print('The Index data is empty!', error)

        # 稳定性数据
        try:

            stab_id_list = config.redis_conn.hkeys(config.HSet_StabQueueName.format(project_id=project_id, 
                                                              object_id=object_id))
            if stab_id_list:
                stab_id = int(max(stab_id_list))
                stability_data = config.redis_conn.hget(config.HSet_StabQueueName.format(project_id=project_id,
                                                                                         object_id=object_id),
                                                       stab_id)
                if stability_data is None:
                    stability_data = []
                elif type(stability_data) is bytes:
                    stability_data = stability_data.decode('utf-8')
                    stability_data = json.loads(stability_data)
                combine_dict["Stab"] = stability_data
        except Exception as error:
            print('The Stability data is empty!', error)

        # 趋势数据
        try:

            tend_id_list = config.redis_conn.hkeys(config.HSet_TendQueueName.format(project_id=project_id, 
                                                              object_id=object_id))
            if tend_id_list:

                tend_id = int(max(tend_id_list))
                tendency_data = config.redis_conn.hget(config.HSet_TendQueueName.format(project_id=project_id,
                                                                                        object_id=object_id),
                                                      tend_id)
                if tendency_data is None:
                    tendency_data = []
                elif isinstance(tendency_data, bytes):
                    tendency_data = literal_eval(tendency_data.decode('utf-8'))
                    tend_Elem = tendency_data["Tend_Elem"]
                    empty_list = []
                    if len(tend_Elem) < 8:
                        for _ in range(8-len(tend_Elem)):
                            tend_Elem.append(empty_list)
                    tend_Pred = tendency_data["Tend_Pred"]
                    if len(tend_Pred) < 8:
                        for _ in range(8-len(tend_Pred)):
                            tend_Pred.append(empty_list)
                    tend_data = dict()
                    tend_data["Tend_Elem"] = tend_Elem
                    tend_data["Tend_Pred"] = tend_Pred
                    tend = json.dumps(tend_data)
                    tendency_data = json.loads(tend)
                combine_dict["Tend"] = tendency_data
        except Exception as error:
            tendency_data = []
            print('The Tendency data is empty!', error)

        # 风险数据
        try:
            risk_id_list = config.redis_conn.hkeys(config.HSet_RiskQueueName.format(project_id=project_id, 
                                                              object_id=object_id))
            if risk_id_list:
                risk_id = int(max(risk_id_list))
                risk_data = config.redis_conn.hget(config.HSet_RiskQueueName.format(project_id=project_id,
                                                                                    object_id=object_id),
                                                  risk_id)
                if risk_data is None:
                    risk_data = []
                elif isinstance(risk_data, bytes):
                    risk_data = risk_data.decode('utf-8')
                    risk_data = json.loads(risk_data)
                combine_dict["Risk"] = risk_data
        except Exception as error:
            print('The risk data is empty', error)

        # 原始数据
        try:
            origial_id_list = config.redis_conn.hkeys(
                config.Hset_OriginalQueueName.format(project_id=project_id, object_id=object_id))
            if origial_id_list:
                ori_id = int(max(origial_id_list))
                original_data = config.redis_conn.hget(config.Hset_OriginalQueueName.format(project_id=project_id,
                                                                                            object_id=object_id),
                                                      ori_id)
                if original_data is None:
                    original_data = []
                elif type(original_data) is bytes:
                    original_data = literal_eval(original_data.decode('utf-8'))
                    # original_data = json.loads(original_data)
                combine_dict["Original"] = original_data
        except Exception as error:
            print('The Orginial data is empty!', error)

        # 优化数据1
        try:
            opt_one_id_list = config.redis_conn.hkeys(
                config.HSet_Opti_One_QueueName.format(project_id=project_id,
                                                      object_id=object_id))
            if opt_one_id_list:
                one_id = int(max(opt_one_id_list))
                opt_one = config.redis_conn.hget(config.HSet_Opti_One_QueueName.format(project_id=project_id,
                                                                                       object_id=object_id),
                                                                                one_id)
                if opt_one is None:
                    opt_one = {}
                elif isinstance(opt_one, bytes):
                    opt_one = opt_one.decode('utf-8')
                    opt_one = json.loads(opt_one)
                combine_dict["Opt_One"] = opt_one
        except Exception as error:
            print('The Optimization data is empty!', error)

        # 优化数据2
        try:
            opt_two_id_list = config.redis_conn.hkeys(
                config.HSet_Opti_Two_QueueName.format(project_id=project_id,
                                                      object_id=object_id))
            if opt_two_id_list:
                opt_tow_id = int(max(opt_two_id_list))
                opt_two = config.redis_conn.hget(config.HSet_Opti_Two_QueueName.format(project_id=project_id,
                                                                                       object_id=object_id),
                                                opt_tow_id)
                if opt_two is None:
                    opt_two = {}
                elif isinstance(opt_two, bytes):
                    opt_two = opt_two.decode('utf-8')
                    opt_two = json.loads(opt_two)
                    combine_dict["Opt_Two"] = opt_two
        except Exception as error:
            print('The  Optimization data of second is empty!', error)

        # 1.存放参数列表
        combine_dict["Parm"] = self.parm_list
        combine_dict["Opt_Parm"] = self.opt_parm_list
        # 2.存放参数中文名称
        combine_dict["Parm_CN"] = self.analysis_parm_cn_list
        combine_dict["Opt_Parm_CN"] = self.opt_parm_cn_list

        # 3.存放参数英文名称
        combine_dict["Parm_EN"] = self.analysis_parm_en_list
        combine_dict["Opt_Parm_EN"] = self.opt_parm_en_list

        try:
            combine_json_data = json.dumps(combine_dict, sort_keys=True)
        except Exception as error:
            print('Combine Error:', error)
        else:
            print('combine_json_data>>>>>>>\n', combine_json_data)
            combine_queue_name = config.HSet_CombineOut_Queue.format(project_id=project_id,
                                                                     object_id=object_id)
            config.redis_conn.hset(combine_queue_name, data_block_id, combine_json_data)

            combine_channel_name = config.Channel_CombineOut_Queue.format(customer_id=self.account_login,
                                                                          project_id=project_id,
                                                                          object_id=object_id)
            config.redis_conn.publish(combine_channel_name, combine_json_data)

            # 判断redis队列中长度是否超过Max_Compute_Queue_Size条，超过以后,进行删除最旧的一条计算数据
            if config.redis_conn.hlen(combine_queue_name ) > config.Max_Compute_Queue_Size:
                        key_list = config.redis_conn.hkeys(combine_queue_name)
                        new_key_list = []
                        for keys in key_list:

                            key_min = keys.decode('utf-8')
                            new_key_list.append(int(key_min))
                        key = min(new_key_list)
                        config.redis_conn.hdel(combine_queue_name , key)
            else:
                pass

            json_body = [
                {
                    "measurement": "Combine",
                    "tags": {
                        "object": "%s:%s" %(project_id,object_id),
                    },
                    # "time": "2017-03-12T22:00:00Z",
                    "fields": {
                        "savetime":str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                        "combine":str(combine_dict)
                    }
                }
            ]
            config.influxdb_conn.write_points(json_body)

    def process_combine(self, sub_organizer_channel):
        """
        :param sub_organizer_channel: 设备对象
        :param parm_list: 参数列表
        :param parm_cn_list: 参数中文名列表
        :return:
        """
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
            thread = threading.Timer(interval=config.Max_Combine_Interval, function=self.combine_data_block,
                                     args=(project_id, object_id, data_block_id))
            thread.run()

    def run(self):
        self.parm_list = []
        self.analysis_parm_cn_list = list()
        self.analysis_parm_en_list = []
        self.opt_parm_en_list = []
        self.opt_parm_cn_list = []

        for parm in self.analysis_script:
            self.parm_list.append(parm[0])
        for parm in self.optimization_script:
            self.opt_parm_list.append(parm[0])
        # 获取中文名
        for parameter in self.analysis_parameter_list:
            self.analysis_parm_cn_list.append(parameter["para_name_cn"])

        for parameter in self.opt_parameter_list:
            self.opt_parm_cn_list.append(parameter["para_name_cn"])

        # 获取英文名
        for parameter in self.analysis_parameter_list:
            self.analysis_parm_en_list.append(parameter["para_name_en"])

        for parameter in self.opt_parameter_list:
            self.opt_parm_en_list.append(parameter["para_name_en"])

        sub_organizer_channel = []
        if self.object_list is not None:
            for object_dict in self.object_list:
                client_id = str(object_dict["object_id"])
                channel_name = config.Channel_OrganizerName.format(project_id=self.project_id,
                                                                   object_id=client_id)
                sub_organizer_channel.append(channel_name)

        self.process_combine(sub_organizer_channel)


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
            combine_daemon = CombineDaemon(project=project_id)
            combine_daemon.set_project_info(project_info)
            combine_daemon.start()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} [start|stop]'.format(sys.argv[0]), file=sys.stderr)
        raise SystemExit(1)
    queue_id = sys.argv[1]
    main(queue_id)
