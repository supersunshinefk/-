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
# Author :ZhangSeyi
# Description : 多个设备mosaic数据获取

import sys
import json
import time
import threading
import sched

import config
from daemon import Daemon
from util.req_gardsdb import dbapi_mosaic

program_type = config.ProgramEnum.pump.name
sched_delay = 5
script_len = 19

class PumpDaemon(Daemon):
    def __init__(self, project=None):
        Daemon.__init__(self, project, program_type)
        self.project_id = None
        self.object_list = None
        self.organizer_script = None
        self.account_login = None
        self.account_pin = None
        self._schedule = sched.scheduler(time.time, time.sleep)

    # 获取indass_boss队列数据
    def set_project_info(self, project_info):
        try:
            # 获取project_id
            self.project_id = project_info["project_id"]

            # 获取object组
            self.object_list = project_info["object"]

            # 获取参数标识列表
            self.organizer_script = project_info["data_organizer"]["script"]
            self.account_login = project_info['account']['account_login']
            self.account_pin = project_info['account']['account_pin']

        except Exception as e:
            print("获取indass_boss数据错误：%s" % e)

            # 获取参数标识整体字符串

    def get_para_reference_name_str(self):

        all_parameter_list = self.organizer_script["all_parameter"]
        para_reference_name_str = ','.join(all_parameter_list)
        return para_reference_name_str

    # 合并多个设备mosaic数据到物联网对象
    def combine_mosaic_data(self, object_id, client_list):
        device_mosaic_dict = dict()

        if len(self.organizer_script["all_parameter"][1]) > script_len:
            for client_id in client_list:
                client_id = client_id.strip()
                mosiac_name = "hset:mosaic:%s" % (client_id)
                key_pref = "%s" % client_id  # 担心多个设备mosaic中存在相同key，故取设备标示对key加前缀

                mosaic_data = config.redis_conn.hgetall(mosiac_name)
                for key in mosaic_data.keys():
                    key_name = "%s_%s" % (key_pref, key.decode())
                    val_json = json.loads(mosaic_data[key])
                    if val_json:
                        device_mosaic_dict[key_name] = val_json["value"]

            device_mosaic_dict["payload_time"] = time.strftime('%Y-%m-%d %H:%M:%S')
            device_mosaic_dict["time"] = time.strftime('%Y-%m-%d %H:%M:%S')

        else:
            for client_id in client_list:
                client_id = client_id.strip()
                mosiac_name = "hset:mosaic:%s" % (client_id)
                key_pref = "A%s" % client_id[-4:]  # 担心多个设备mosaic中存在相同key，故取设备标示对key加前缀

                mosaic_data = config.redis_conn.hgetall(mosiac_name)
                for key in mosaic_data.keys():
                    key_name = "%s_%s" % (key_pref, key.decode())
                    val_json = json.loads(mosaic_data[key])
                    if val_json:
                        device_mosaic_dict[key_name] = val_json["value"]

            device_mosaic_dict["payload_time"] = time.strftime('%Y-%m-%d %H:%M:%S')
            device_mosaic_dict["time"] = time.strftime('%Y-%m-%d %H:%M:%S')

        # 构建队列名
        pump_queue_name = config.List_PumpQueueName % (self.project_id, object_id)
        print(">>>>device_mosaic_dict - >>>>@@@", device_mosaic_dict)
        print(">>>pump_queue_name:", pump_queue_name, device_mosaic_dict)
        config.redis_conn.lpush(pump_queue_name, json.dumps(device_mosaic_dict))

        # self._schedule.enter(sched_delay, 2, self.combine_mosaic_data, argument=(object_id, client_list,))

        _thread = threading.Timer(interval=sched_delay, function=self.combine_mosaic_data,
                                  args=(object_id, client_list))
        _thread.start()

    def run(self):
        # jobqueue = Queue.Queue()
        # 遍历多台设备对象
        for object_dict in self.object_list:
            # 获取数据源数据
            # 获取设备编号
            object_id = object_dict["object_id"]
            object_template = object_dict["template"]
            client_list = object_template.split(";")

            # schedule.every(5).seconds.do(self.combine_mosaic_data, (object_id, client_list))
            # self._schedule.enter(sched_delay, 2, self.combine_mosaic_data, argument=(object_id, client_list,))
            # self._schedule.run()

            # 开启线程处理发送数据到队列的任务
            _thread = threading.Timer(interval=sched_delay, function=self.combine_mosaic_data,
                                      args=(object_id, client_list))
            _thread.start()



###############################################################################
###############################################################################
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

            pump_daemon = PumpDaemon(project=project_id)
            pump_daemon.set_project_info(project_info)
            pump_daemon.start()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} [start|stop]'.format(sys.argv[0]), file=sys.stderr)
        raise SystemExit(1)

    queue_id = sys.argv[1]
    main(queue_id)
