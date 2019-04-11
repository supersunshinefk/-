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
# Author : FangKun
# Description : message, 异常消失提示


# encoding = utf-8
import re
import redis
import threading
import config
import sys
import time
from ast import literal_eval
from daemon import Daemon
import json
import requests
import urllib

program_type = config.ProgramEnum.message.name


class MessageDaemon(Daemon):
    def __init__(self, project=None):
        Daemon.__init__(self, project, program_type)
        self.object_list = None
        self.project_id = None
        self.message_script = None
        self.username = None
        self.password = None

    def set_project_info(self, project_info):
        try:
            # 获取对象列表以及项目id
            self.object_list = project_info["object"]
            self.project_id = project_info["project_id"]
            # 获取登录名
            self.username = project_info['account']['account_login']
            # 获取登录密码
            self.password = project_info['account']['account_pin']
            # 获取boss中message表中的脚本script
            self.message_script = project_info["message_out"]["script"]

        except Exception as e:
            print("获取设备数据出错：%s" % e)

    # 解析message_script脚本中的运算符
    # data1 表示当前实际风险值/稳定性值
    # data2 表示设定的风险值/稳定值  如：风险为90
    def parse_symbol(self, name, symbol, data1, data2, output):
        if symbol == 'equal':
            if abs(data1) == data2:
                return output
            else:
                print('当前{0}为{1}不等于您设定的阈值{2}'.format(name, data1, data2))
                return 0
        if symbol == 'inequal':
            if abs(data1) != data2:
                return output
            else:
                print('当前{0}为{1}等于您设定的阈值{2}'.format(name, data1, data2))
                return 0
        if symbol == 'greater':
            if abs(data1) > data2:
                return output
            else:
                print('当前{0}为{1}小于您设定的阈值{2}'.format(name, data1, data2))
                return 0
        if symbol == 'less':
            if abs(data1) < data2:
                return output
            else:
                print('当前{0}为{1}大于您设定的阈值{2}'.format(name, data1, data2))
                return 0

    def message_data_block(self, project_id, object_id):
        combine_queue_name = config.HSet_CombineOut_Queue.format(project_id=project_id,
                                                                 object_id=object_id)
        print(">>>开始时间", time.time())
        while True:
            key_list = config.redis_conn.hkeys(combine_queue_name)
            if key_list:
                break
        max_key = max(key_list)
        while True:
            combine_data = config.redis_conn.hget(combine_queue_name, max_key)
            if combine_data:
                break
            else:
                print("combine 获取数据为空！！！")
                time.sleep(5)
        combine_data = json.loads(combine_data.decode('utf-8'))
        # message 消息处理
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        for script in self.message_script:
            # # 指数值
            # if script[0] == "Indx":
            #     symbol = re.split(r'@', script[1])[1]
            #     data1 = combine_data['Indx']['Indx']
            #     data2 = script[2]
            #     output = script[3]
            #     message = self.parse_symbol(script[0], symbol, data1, data2, output)
            #     if message != 0:
            #         # 拼接data数据包用于发送
            #         stab_data = dict()
            #         stab_data['message_name'] = self.username + '稳定性提醒 (测试)'
            #         stab_data['date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            #         stab_data['category'] = '稳定性'
            #         stab_data['unit'] = 'indass'
            #         stab_data['messenger'] = object_id
            #         stab_data['description'] = '设备' + str(project_id) + str(message)
            #         stab_data['customer_id'] = ''
            #         self.requests_post(stab_data)
            # 指数梯度上阈值
            if script[0] == "Grad":
                symbol = re.split(r'@', script[2])[1]
                data1 = combine_data['Indx']['Grad']
                data2 = script[3]
                output = script[5]
                message = self.parse_symbol(script[0], symbol, data1, data2, output)
                if message != 0:
                    # 拼接data数据包用于发送
                    Indx_Grad_data = dict()
                    Indx_Grad_data['message_name'] = self.username + '指数梯度上阈值提醒 (测试)'
                    Indx_Grad_data['date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    Indx_Grad_data['category'] = '指数梯度'
                    Indx_Grad_data['unit'] = 'indass'
                    Indx_Grad_data['messenger'] = object_id
                    Indx_Grad_data['description'] = '设备' + str(project_id) + str(message)
                    Indx_Grad_data['customer_id'] = ''
                    self.requests_post(Indx_Grad_data)
            # 指数累积效应上阈值
            if script[0] == "Cumu":
                symbol = re.split(r'@', script[2])[1]
                data1 = combine_data['Indx']['Cumu']
                data2 = script[3]
                output = script[5]
                message = self.parse_symbol(script[0], symbol, data1, data2, output)
                if message != 0:
                    # 拼接data数据包用于发送
                    Indx_Cumu_data = dict()
                    Indx_Cumu_data['message_name'] = self.username + '指数累积效应上阈值提醒 (测试)'
                    Indx_Cumu_data['date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    Indx_Cumu_data['category'] = '指数累积效应'
                    Indx_Cumu_data['unit'] = 'indass'
                    Indx_Cumu_data['messenger'] = object_id
                    Indx_Cumu_data['description'] = '设备' + str(project_id) + str(message)
                    Indx_Cumu_data['customer_id'] = ''
                    self.requests_post(Indx_Cumu_data)
            # # 趋势
            if script[0] == "Tend_Elem":
                symbol = re.split(r'@', script[2])[1]
                for Parm_CN in combine_data['Parm_CN']:
                    if Parm_CN == script[4]:
                        index = combine_data['Parm_CN'].index(Parm_CN)
                        for param_each_value in combine_data['Tend']['Tend_Elem'][index]:
                            data1 = param_each_value
                            data2 = script[3]
                            output = script[5]
                            message = self.parse_symbol(script[0], symbol, data1, data2, output)
                            if message != 0:
                                # 拼接data数据包用于发送
                                stab_data = dict()
                                stab_data['message_name'] = self.username + script[1]
                                stab_data['date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                                stab_data['category'] = '趋势'
                                stab_data['unit'] = 'indass'
                                stab_data['messenger'] = object_id
                                stab_data['description'] = '设备' + str(project_id) + str(message)
                                stab_data['customer_id'] = ''
                                self.requests_post(stab_data)
                                break
            # #######################################################################################
            # 风险
            if script[0] == "Risk":
                symbol = re.split(r'@', script[2])[1]
                data1 = combine_data['Risk']['Risk']
                data2 = script[3]
                output = script[5]
                message = self.parse_symbol(script[0], symbol, data1, data2, output)
                if message != 0:
                    # 拼接data数据包用于发送
                    risk_data = dict()
                    risk_data['message_name'] = self.username + script[1]
                    risk_data['date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    risk_data['category'] = '风险'
                    risk_data['unit'] = 'indass'
                    risk_data['messenger'] = object_id
                    risk_data['description'] = '设备' + str(project_id) + str(message)
                    risk_data['customer_id'] = ''
                    self.requests_post(risk_data)
            # 稳定性
            if script[0] == "Stab":
                symbol = re.split(r'@', script[2])[1]
                data1 = combine_data['Stab']['Indx']
                data2 = script[3]
                output = script[5]
                message = self.parse_symbol(script[0], symbol, data1, data2, output)
                if message != 0:
                    # 拼接data数据包用于发送
                    stab_data = dict()
                    stab_data['message_name'] = self.username + script[1]
                    stab_data['date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    stab_data['category'] = '稳定性'
                    stab_data['unit'] = 'indass'
                    stab_data['messenger'] = object_id
                    stab_data['description'] = '设备' + str(project_id) + str(message)
                    stab_data['customer_id'] = ''
                    self.requests_post(stab_data)

    def process_message(self, sub_combine_channel):
        ps = config.redis_conn.pubsub()
        ps.subscribe(sub_combine_channel)
        for item in ps.listen():
            sub_type = item["type"]
            if sub_type == "subscribe":
                continue
            data_block_id = item["data"]
            if isinstance(data_block_id, bytes):
                data_block_id = data_block_id.decode("utf-8")
            else:
                data_block_id = str(data_block_id)

            channel = item["channel"].decode("utf-8")
            channel_list = channel.split("/")
            project_id = channel_list[2]
            object_id = channel_list[3]
            thread = threading.Timer(interval=config.Max_Combine_Interval, function=self.message_data_block,
                                     args=(project_id, object_id))
            thread.start()

    def requests_post(self, data):
        # config.py  login_url = 'http://mixlinker.mixiot.top:8008/api/login'
        login_url = config.login_url
        # config.py  login_param = {'username': 'admin', 'password': 'mix123456', 'system': 'PRO'}
        login_param = config.login_param
        try:
            # 登录获取token,用于拼接requests_post中的headers
            r1 = requests.post(login_url, data=login_param)
        except Exception as err:
            print(err)
        # print('r1.text>>>>', r1.text)
        # print('r1.status_code>>>>>', r1.status_code)
        dic1 = json.loads(r1.text)
        token = dic1["data"]["token"]
        # 接入API
        # url需要在config.py文件中配置 url = "http://mixlinker.mixiot.top:8008/api/message/add"
        url = config.url
        headers = {
            'Authorization': 'Bearer ' + token
        }
        result = requests.post(url, headers=headers, data=data)
        result = result.content
        print(json.loads(result.decode('utf-8')))

    def run(self):
        sub_combine_channel = []
        if self.object_list is not None:
            for object_dict in self.object_list:
                client_id = str(object_dict["object_id"])

                channel_name = config.Channel_CombineOut_Queue.format(customer_id=self.username,
                                                                      project_id=self.project_id,
                                                                      object_id=client_id)
                sub_combine_channel.append(channel_name)
            # print('*'*30)
            # print('sub_combine_channel', sub_combine_channel)
            self.process_message(sub_combine_channel)


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
            message_daemon = MessageDaemon(project=project_id)
            message_daemon.set_project_info(project_info)
            message_daemon.start()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} [start|stop]'.format(sys.argv[0]), file=sys.stderr)
        raise SystemExit(1)
    queue_id = sys.argv[1]
    main(queue_id)
