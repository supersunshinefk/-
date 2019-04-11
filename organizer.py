#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
# Author :Fangkun & ZhangSeyi
# Description : 数据组织器

__AUTHOR__ = 'fangkun'
__VERSION__ = '1.0.0'
__DATE_CREATED__ = '2018-4-10 14:35:25'
__DATE_MODIFIED__ = '2018-4-10 14:35:25'

import json
import re
import sys
import threading
from io import StringIO
import config
from daemon import Daemon
import numpy as np
import pandas as pd
import time
from ast import literal_eval
import compute_handle
import numpy
import copy



program_type = config.ProgramEnum.organizer.name

sched_delay = config.sched_delay  ##设置取数据时间

class OrganizerDaemon(Daemon):
    def __init__(self, project=None):
        Daemon.__init__(self, project, program_type)
        self.project_id = None
        self.object_list = None
        self.data_organizer = None

    # 获取indass_boss队列数据
    def set_project_info(self, project_info):
        self.project_id = project_info["project_id"]
        # 获取object组
        self.object_list = project_info["object"]
        # 获取组织器脚本
        self.data_organizer = project_info["data_organizer"]

    def organizer_data(self, client_id):

        # 获取组织器脚本
        analysis_parameter_list = self.data_organizer["script"]["analysis_parameter"]
        optimization_parameter_list = self.data_organizer["script"]["optimization_parameter"]

        # 获取分析参数l符号名称
        analysis_l_list = list()
        # 获取分析计算参数运算符
        analysis_symbol_list = list()
        # 获取分析计算的参数列表
        analysis_compute_list = list()
        # 分析参数变量名列表
        analysis_variate_list = list()
        for i in analysis_parameter_list:
            analysis_variate_list.append(i[0])
            if not i[1].startswith("@"):
                analysis_l_list.append(i[1])
            else:
                analysis_compute_list.append(i[-1])
                analysis_symbol_list.append(i[1].strip("@"))

        # 获取优化参数l符号名称
        optimization_l_list = list()
        # 获取优化计算参数运算符
        optimization_symbol_list = list()
        # 获取分析计算的参数列表
        optimization_compute_list = list()
        # 分析参数变量名列表
        optimization_variate_list = list()
        for i in optimization_parameter_list:
            optimization_variate_list.append(i[0])
            if not i[1].startswith("@"):
                optimization_l_list.append(i[1])
            else:
                optimization_compute_list.append(i[-1])
                optimization_symbol_list.append(i[1].strip("@"))

        # 定义一个列表容器，放置每个分析参数的config.llen条数据
        analysis_all_list = list()
        for i in range(len(analysis_l_list)):
            analysis_all_list.append([])

        # 定义一个列表容器，放置每个优化参数的300条数据
        optimization_all_list = list()
        for i in range(len(optimization_l_list)):
            optimization_all_list.append([])
##############################################################################################
        # 定义一个列表容器放置分析计算的参数数据
        analysis_compute_all_list = list()
        for i in range(len(analysis_compute_list)):
            analysis_compute_all_list.append([])

        # 定义一个列表容器放置优化计算的参数数据
        optimization_compute_all_list = list()
        for i in range(len(optimization_compute_list)):
            optimization_compute_all_list.append([])

        # 分析参数容器
        for i in analysis_compute_all_list:
            for j in range(len(analysis_compute_list[analysis_compute_all_list.index(i)])):
                i.append([])

        # 优化参数容器
        for i in optimization_compute_all_list:
            for j in range(len(optimization_compute_list[optimization_compute_all_list.index(i)])):
                i.append([])
################################################################################################
        # 定义时间列表容器
        time_list = list()

        # 定义一个变量，判断是否为第一次分析参数计算
        analysis_is_first = True
        # 定义一个变量，判断是否为第一次优化参数计算
        optimization_is_first = True
        # 定义一个列表，存放第一次分析参数计算的数据
        analysis_instance_list = list()
        # 定义一个列表，存放第一次优化参数计算的数据
        optimization_instance_list = list()

        analysis_receive_data = None
        optimization_receive_data = None

        my_num = 1
        our_num = 1

        filter_queue_name = config.List_FilterQueueName % (self.project_id, client_id)
        # res = config.redis_conn.brpop(filter_queue_name)[1]
        while True:
            llen = config.redis_conn.llen(filter_queue_name)
            # time.sleep((config.llen-llen)*sched_delay)
            if llen > config.llen:
                while True:
                    res = config.redis_conn.brpop(filter_queue_name)[1]
                    if not res:
                        print('Get Filter Data Error!')
                        continue
                    res = res.decode("utf-8")
                    organizer_data = json.loads(res)
                    # 如果数据为空， 跳到下一个循环
                    if not organizer_data:
                        print("The organizer receives data to be empty！")
                        continue

                    # 获取每个分析参数的每条值放置到各自的列表中
                    for i, j in zip(analysis_l_list, analysis_all_list):
                        j.append(organizer_data[i])

                    # 获取每个优化参数的每条值放置到各自的列表中
                    for i, j in zip(optimization_l_list, optimization_all_list):
                        j.append(organizer_data[i])

                    # 遍历分析参数大容器
                    for i, j in zip(analysis_compute_list, analysis_compute_all_list):
                        for m, n in zip(i, j):
                            n.append(organizer_data[m])

                    # 遍历优化参数大容器
                    for i, j in zip(optimization_compute_list, optimization_compute_all_list):
                        for m, n in zip(i, j):
                            n.append(organizer_data[m])

                    # 时间格式转为时间戳
                    new_time_str = int(time.mktime(time.strptime(organizer_data["time"], '%Y-%m-%d %H:%M:%S')))

                    time_list.append(new_time_str)

                    # 分析参数计算
                    if len(analysis_all_list[0]) > config.llen-1:

                        # 所有分析参数值赋值给一个变量(没时间)
                        analysis_all_list_save = copy.deepcopy(analysis_all_list)
                        # 所有计算参数的相关参数值赋值给一个变量(没时间)
                        analysis_compute_all_list_save = copy.deepcopy(analysis_compute_all_list)

                        # 时间戳转为数字
                        new_time_list = []
                        for i in time_list:
                            num = (i - time_list[0]) / sched_delay
                            new_time_list.append(num)
                        # 把时间添加到总列表里面（所有的分析参数config.llen个值的组合）
                        analysis_all_list_save.insert(0, new_time_list)

                        # 1. 有计算参数
                        if len(analysis_compute_list) > 0:

                            if analysis_is_first:
                                print("Analysis the parameter data the first times")
                                for i, j in zip(analysis_symbol_list, analysis_compute_all_list_save):

                                    j.insert(0, new_time_list)
                                    j_mat = pd.DataFrame(np.mat(j))
                                    index_list = list(range(len(j)))[1:]
                                    data = compute_handle.organize_dataset_first(j_mat, i, index_list)
                                    # 把计算出来的整条config.llen个数组添加到整体队列中
                                    analysis_all_list_save.append(data)
                                    # 存放第一次数据
                                    analysis_instance_list.append(data)

                                # 把计算出来的参数列表保存给下一轮循环
                                analysis_receive_data = copy.deepcopy(analysis_instance_list)

                                # 保存一个发送字典
                                new_analysis_all_list_save = copy.deepcopy(analysis_all_list_save)
                                # 1. 把数据的第一行时间去掉
                                new_analysis_all_list_save.pop(0)
                                # 2. 构造发送数据列表
                                analysis_send_list = list()
                                for j in new_analysis_all_list_save:
                                    analysis_send_list.append(j[config.llen-1])
                                result = json.dumps(analysis_all_list_save)
                                #
                                organizer_analysis_param_queue_name = config.Hset_Organizer_Analysis_QueueName.format(
                                    project_id=self.project_id, object_id=client_id)
                                self.analysis_send_data(client_id, new_time_str, result, analysis_send_list, organizer_analysis_param_queue_name)

                                # 处理数据给下一次循环
                                # 1．总的分析参数列表处理
                                # 分析各参数列表长度删除第一个数据，即最旧的那个数据
                                for param_list in analysis_all_list:
                                    param_list.pop(0)
                                # 2.删除计算完以后config.llen条数据的第一个(不带时间)
                                for param_list in analysis_receive_data:
                                    param_list.pop(0)
                                # 3. 处理要计算分析参数相关的参数的列表(不带时间)
                                for m in analysis_compute_all_list:
                                    for n in m:
                                        n.pop(0)

                                analysis_is_first = False

                            else:

                                my_num += 1
                                print("Analysis the parameter data %d times...." % my_num)
                                for i, j, k in zip(analysis_symbol_list, analysis_compute_all_list_save, analysis_receive_data):
                                    j.insert(0, new_time_list)
                                    index_list = list(range(len(j)))[1:]
                                    data = compute_handle.organize_dataset(np.mat(j), i, index_list)
                                    k.append(data)

                                # 构造发送数据:总的分析参数列表
                                analysis_all_list_save += analysis_receive_data

                                # 保存一个发送字典
                                new_analysis_all_list_save = copy.deepcopy(analysis_all_list_save)
                                # 1. 把数据的第一行时间去掉
                                new_analysis_all_list_save.pop(0)
                                # 2. 构造发送数据字典
                                analysis_send_list = list()
                                for j in new_analysis_all_list_save:
                                    analysis_send_list.append(j[config.llen-1])
                                result = json.dumps(analysis_all_list_save)
                                organizer_analysis_param_queue_name = config.Hset_Organizer_Analysis_QueueName.format(project_id=self.project_id, object_id=client_id)
                                self.analysis_send_data(client_id, new_time_str, result, analysis_send_list, organizer_analysis_param_queue_name)
                                print('output ->>>>', result)

                                # 处理数据给下一次循环
                                # 1．总的分析参数列表处理
                                # 分析各参数列表长度删除第一个数据，即最旧的那个数据
                                for param_list in analysis_all_list:
                                    param_list.pop(0)
                                # 2.删除计算完以后config.llen条数据的第一个(不带时间)
                                for param_list in analysis_receive_data:
                                    param_list.pop(0)
                                # 3. 处理要计算分析参数相关的参数的列表(不带时间)
                                for m in analysis_compute_all_list:
                                    for n in m:
                                        n.pop(0)
                                # time.sleep(5)

                        # 2. 没计算参数
                        else:
                            my_num += 1
                            print("Analysis the parameter data %d times..." % my_num)
                            # 1.处理发送数据
                            # 1.1．　构造分析参数列表数据
                            result = json.dumps(analysis_all_list_save)

                            # 1.2．构造发送字典
                            new_analysis_all_list_save = copy.deepcopy(analysis_all_list_save)
                            # 去掉第一行时间行
                            new_analysis_all_list_save.pop(0)
                            # 2. 构造发送数据字典
                            analysis_send_list = list()
                            for j in new_analysis_all_list_save:
                                analysis_send_list.append(j[config.llen-1])

                            # 2.发送数据到redis
                            organizer_analysis_param_queue_name = config.Hset_Organizer_Analysis_QueueName.format(
                                project_id=self.project_id, object_id=client_id)
                            self.analysis_send_data(client_id, new_time_str, result, analysis_send_list, organizer_analysis_param_queue_name)

                            # 3．处理数据给下一次循环
                            # 分析各参数列表长度删除第一个数据，即最旧的那个数据
                            for param_list in analysis_all_list:
                                param_list.pop(0)



        #################################################################################################################
                    # 优化参数计算
                    if len(optimization_all_list[0]) > config.llen-1:
                        # 把optimization_all_list数据赋值给optimization_all_list_save变量去计算
                        optimization_all_list_save = copy.deepcopy(optimization_all_list)
                        # 把optimization_compute_all_list数据赋值给optimization_compute_all_list_save变量去计算
                        optimization_compute_all_list_save = copy.deepcopy(optimization_compute_all_list)
                        # 时间戳转为数字
                        new_time_list = []
                        for i in time_list:
                            num = (i - time_list[0]) / sched_delay
                            new_time_list.append(num)
                        # 把时间添加到总列表里面
                        optimization_all_list_save.insert(0, new_time_list)

                        # 1. 有计算的优化参数
                        if len(optimization_compute_list) > 0:

                            if optimization_is_first:
                                print("Optimize the parameter data the first times！！")
                                for i, j in zip(optimization_symbol_list, optimization_compute_all_list_save):
                                    j.insert(0, new_time_list)
                                    # j_mat = np.mat(j)
                                    index_list = list(range(len(j)))
                                    data = compute_handle.organize_dataset_first(j, i, index_list)
                                    # 存放第一次数据
                                    optimization_instance_list.append(data)

                                    optimization_all_list_save.append(data)
                                # 把计算出来的参数列表保存给下一轮循环
                                optimization_receive_data = copy.deepcopy(optimization_instance_list)

                                # 保存一个发送字典
                                new_optimization_all_list_save = copy.deepcopy(optimization_all_list_save)
                                # 1. 把数据的第一行时间去掉
                                new_optimization_all_list_save.pop(0)
                                # # 2. 构造发送数据字典
                                # optimization_send_dict = dict()
                                # for i, j in zip(analysis_variate_list, new_optimization_all_list_save):
                                #     optimization_send_dict[i] = j[0]
                                result = json.dumps(optimization_all_list_save)
                                # print(result)
                                organizer_optimization_param_queue_name = config.Hset_Organizer_Optimization_QueueName.format(
                                    project_id=self.project_id, object_id=client_id)
                                self.optimization_send_data(client_id, new_time_str, result,
                                               organizer_optimization_param_queue_name)

                                # 处理数据给下一次循环
                                # 1．总的优化参数列表处理
                                # 分析各参数列表长度删除第一个数据，即最旧的那个数据
                                for param_list in optimization_all_list:
                                    param_list.pop(0)
                                # 2.删除计算完以后config.llen条数据的第一个(不带时间)
                                for param_list in optimization_receive_data:
                                    param_list.pop(0)
                                # 3. 处理要计算优化参数相关的参数的列表(不带时间)
                                for m in optimization_compute_all_list:
                                    for n in m:
                                        n.pop(0)

                                optimization_is_first = False

                            else:
                                our_num += 1
                                print("Optimize the parameter data %d times...." % our_num)
                                for i, j, k in zip(optimization_symbol_list, optimization_compute_all_list_save, optimization_receive_data):
                                    j.insert(0, new_time_list)
                                    index_list = list(range(len(j)))[1:]
                                    data = compute_handle.organize_dataset(np.mat(j), i, index_list)
                                    k.append(data)

                                # 构造发送数据:总的优化参数列表
                                optimization_all_list_save += optimization_receive_data

                                # 保存一个发送字典
                                new_optimization_all_list_save = copy.deepcopy(optimization_all_list_save)
                                # 1. 把数据的第一行时间去掉
                                new_optimization_all_list_save.pop(0)
                                # 2. 构造发送数据字典
                                optimization_send_dict = dict()
                                for i, j in zip(optimization_variate_list, new_optimization_all_list_save):
                                    optimization_send_dict[i] = j[0]
                                result = json.dumps(optimization_all_list_save)
                                organizer_optimization_param_queue_name = config.Hset_Organizer_Optimization_QueueName.format(
                                    project_id=self.project_id, object_id=client_id)
                                self.optimization_send_data(client_id, new_time_str, result,
                                               organizer_optimization_param_queue_name)

                                # 处理数据给下一次循环
                                # 1．总的优化参数列表处理
                                # 优化各参数列表长度删除第一个数据，即最旧的那个数据
                                for param_list in optimization_all_list:
                                    param_list.pop(0)
                                # 2.删除计算完以后300条数据的第一个(不带时间)
                                for param_list in optimization_receive_data:
                                    param_list.pop(0)
                                # 3. 处理要计算优化数相关的参数的列表(不带时间)
                                for m in optimization_compute_all_list:
                                    for n in m:
                                        n.pop(0)
                                # print("resss", result)
                                # time.sleep(5)
                        #2. 没计算参数
                        else:
                            # 1.处理发送数据
                            # 1.1．　构造优化参数列表数据
                            result = json.dumps(optimization_all_list_save)
                            # 1.2．构造发送字典
                            new_optimization_all_list_save = copy.deepcopy(optimization_all_list_save)
                            # 去掉第一行时间行
                            new_optimization_all_list_save.pop(0)
                            optimization_send_dict = dict()
                            for i, j in zip(analysis_variate_list, new_optimization_all_list_save):
                                optimization_send_dict[i] = j[0]

                            # 2.发送数据到redis
                            organizer_optimization_param_queue_name = config.Hset_Organizer_Optimization_QueueName.format(
                                project_id=self.project_id, object_id=client_id)
                            self.optimization_send_data(client_id, new_time_str, result,
                                           organizer_optimization_param_queue_name)

                            print("output ->>>>", result)
                            # 3．处理数据给下一次循环
                            # 分析各参数列表长度删除第一个数据，即最旧的那个数据
                            for param_list in optimization_all_list:
                                param_list.pop(0)
                        # 删除最旧的时间戳
                        time_list.pop(0)

    def analysis_send_data(self, client_id, new_time_str, data, variate_dict, name):
        # 定义分发的队列名称!!!!!!!!!!!!(config配置文件需要更改！！！)


        config.redis_conn.hset(name, new_time_str, data)

        if config.redis_conn.hlen(name) > config.Max_Organzier_Queue_Size:
            key_list = config.redis_conn.hkeys(name)
            key = min(key_list)
            config.redis_conn.hdel(name, key)

            data_block_id_queue_name = config.Incr_OrganizerName.format(project_id=self.project_id,
                                                                        object_id=client_id)
            channel_organizer_queue_name = config.Channel_OrganizerName.format(project_id=self.project_id,
                                                                               object_id=client_id)
            data_block_id = config.redis_conn.incr(data_block_id_queue_name)
            print(
                ">>{0},project_id={1},object_id={2},data_block_id={3}".format(program_type, self.project_id,
                                                                              client_id,
                                                                              data_block_id))
            config.redis_conn.publish(channel_organizer_queue_name, data_block_id)
            orgin_data = config.Hset_OriginalQueueName.format(project_id=self.project_id,
                                                              object_id=client_id)
            config.redis_conn.hset(orgin_data, data_block_id, variate_dict)

            if config.redis_conn.hlen(orgin_data) > config.ORGANIZER_QUEUE_LINE_MAX:
                key_list_ori = config.redis_conn.hkeys(orgin_data)
                new_key_list = []
                for keys in key_list_ori:

                    key_min = keys.decode('utf-8')
                    new_key_list.append(int(key_min))
                key = min(new_key_list)
                config.redis_conn.hdel(orgin_data, key)
            else:
                pass
            json_body = [
                {
                    "measurement": "Original",
                    "tags": {
                        "object": "%s:%s" %(self.project_id,client_id),
                    },
                    # "time": "2017-03-12T22:00:00Z",
                    "fields": {
                        "savetime":str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                        "original":str(variate_dict)
                    }
                }
            ]
            config.influxdb_conn.write_points(json_body) 
            print('Over Time!', time.time())
    def optimization_send_data(self, client_id, new_time_str, data, name):
        # 定义分发的队列名称!!!!!!!!!!!!(config配置文件需要更改！！！)

        config.redis_conn.hset(name, new_time_str, data)

        if config.redis_conn.hlen(name) > config.Max_Organzier_Queue_Size:
            key_list = config.redis_conn.hkeys(name)
            key = min(key_list)
            config.redis_conn.hdel(name, key)
            data_block_id_queue_name = config.Incr_OptimiizationName.format(project_id=self.project_id,
                                                                        object_id=client_id)
            channel_organizer_queue_name = config.Channel_OptimiizationName.format(project_id=self.project_id,
                                                                               object_id=client_id)
            data_block_id = config.redis_conn.incr(data_block_id_queue_name)
            print(
                ">>{0},project_id={1},object_id={2},data_block_id={3}".format(program_type, self.project_id,
                                                                              client_id,
                                                                              data_block_id))
            config.redis_conn.publish(channel_organizer_queue_name, data_block_id)
            # orgin_data = config.Hset_OriginalQueueName.format(project_id=self.project_id, object_id=client_id)
            # config.redis_conn.hset(orgin_data, data_block_id, variate_dict)



    def run(self):
        thread_list = []
        for object_dict in self.object_list:
            # 获取数据源数据
            # 获取设备编号
            client_id = str(object_dict["object_id"])

            # 构建队列名
            thread = threading.Thread(target=self.organizer_data, args=(client_id,))
            thread_list.append(thread)

        # 设置守护主进程
        for t in thread_list:

            t.setDaemon(True)
            t.start()
        t.join()


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

            organizer_daemon = OrganizerDaemon(project=project_id)
            organizer_daemon.set_project_info(project_info)
            organizer_daemon.start()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} [start|stop]'.format(sys.argv[0]), file=sys.stderr)
        raise SystemExit(1)

    queue_id = sys.argv[1]
    main(queue_id)
