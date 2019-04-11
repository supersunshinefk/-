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
# Description : pump 数据转发

import datetime
import sys
import json
import time
import threading
import requests
import config
import ast
from daemon import Daemon

program_type = config.ProgramEnum.pump.name


class PumpDaemon(Daemon):
    def __init__(self, project=None):
        Daemon.__init__(self, project, program_type)
        self.project_id = None
        self.object_list = None
        self.account_login = None
        self.account_pin = None
        self.organizer_script = None



    # 获取indass_boss队列数据
    def set_project_info(self, project_info):
        try:
            # 获取project_id
            self.project_id = project_info["project_id"]
            # 获取object组
            self.object_list = project_info["object"]
            # 获取组织器脚本
            self.organizer_script = project_info["data_organizer"]["script"]
            self.account_login = project_info['account']['account_login']
            self.account_pin = project_info['account']['account_pin']
            self.data_organizer = project_info["data_organizer"]

        except Exception as e:
            print("Failed to get the project info!：%s" % e)

    def get_fidis_keys_list(self):
        keys_list = list()  
        script = dict(self.organizer_script)
        for i in script["all_parameter"]:
            keys_list.append(i)  
        key = json.dumps(keys_list) 
        return key
    # 获取历史数据
    def get_first_time_data(self, equipment_id, page_num, page_size, keys_list, name):

        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {
            "equipment_id": equipment_id,
            "page_size": page_size,
            "page": page_num,
            "keys":  keys_list,
            "end_time": now_time
        }
        try:
            response = requests.post(config.APIX_URL, data=data)
            receive_data = ast.literal_eval(response.text)
        except Exception as e:
            print("Requested apix history's data was wrong!")
            return 
        code = receive_data["code"]
        if code != 200:
            print("Get fidis data is failed!")
            return
        result_list = receive_data["result"]["result"]
        if len(result_list) == 0:
            print("Get fidis data is null!")
            return
        result_list = receive_data["result"]["result"]

        data_time_list = list()
        all_data_list = list()
        for result in result_list:
            value_dict = result["value"]
            all_data_list.append(value_dict)
            timeArray = time.strptime(result["time"], "%Y-%m-%d %H:%M:%S")
            timeStamp = int(time.mktime(timeArray))
            data_time_list.append(timeStamp)

        all_index_list = list()
        for i in range(config.DATA_COUNT):
            interval = i * config.TIME_INTERVAL
            num = data_time_list[0] - interval
            for m, n in zip(data_time_list[:-2], data_time_list[1:]):
                if (num <= m) and (num >= n):
                    all_index_list.append(data_time_list.index(m))
                    break
        new_all_data_list = list()
        # 获取当前北京时间的时间戳
        int_time = int(time.time()) + 8 * 60 * 60
        for i in all_index_list:
            str_time = datetime.datetime.utcfromtimestamp(int_time)
            str_time = str_time.strftime('%Y-%m-%d %H:%M:%S')
            all_data_list[i]["time"] = str_time
            json_data = json.dumps(all_data_list[i])
            new_all_data_list.append(json_data)
            int_time -= config.TIME_INTERVAL

        for i in new_all_data_list[::-1]:
            print(i)
            config.redis_conn.lpush(name, i)

    # 获取数据，处理数据
    def get_other_time_data(self, equipment_id, page_num, page_size, keys_list, now_time, name):

        data = {
            "equipment_id": equipment_id,
            "page_size": page_size,
            "page": page_num,
            "keys": keys_list
        }
        try:
            response = requests.post(config.APIX_URL, data=data)
            receive_data = ast.literal_eval(response.text)
        except Exception as e:
            print("Requested apix data is wrong")
            return 
        code = receive_data["code"]
        if code != 200:
            print("Get fidis data is failed")
            return

        result_list = receive_data["result"]["result"]
        if len(result_list) == 0:
            print("Get fidis data is null")
            return

        for result in result_list[::-1]:
            result["value"]["time"] = now_time
            json_data = json.dumps(result["value"])
            print(json_data)
            # 把数据放到redis队列
            config.redis_conn.lpush(name, json_data)

    # 定义发送数据到队列的任务
    def send_data(self, client_id, name):

        is_first = True
        num = 2
        while True:
            keys_list = self.get_fidis_keys_list()
            if is_first:
                print(">>The first of times to store data")
                page_num = 1
                page_size = config.PAGE_SIZE
                self.get_first_time_data(client_id, page_num, page_size, keys_list, name)
                is_first = False
                time.sleep(config.TIME_INTERVAL)
                continue
            now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            page_num = 1
            page_size = 1
            print(">> %d of times to store data" % num)
            start_time = time.time()
            self.get_other_time_data(client_id, page_num, page_size, keys_list, now_time, name)
            end_time = time.time()
            elapsed_time = end_time-start_time
            # 设置时间延迟再次获取数据
            sleep_time = config.TIME_INTERVAL - elapsed_time
            if sleep_time < 0:
                print("The request time to fetch data is greater than the time interval")
                num += 1
                continue
            time.sleep(sleep_time)
            num += 1

    def run(self):
        # 定义一个列表存放线程
        thread_list = []
        # 遍历多台设备对象
        for object_dict in self.object_list:
            # 获取数据源数据
            # 获取设备编号
            client_id = object_dict["object_id"]
            client_id = str(client_id)

            # 构建队列名
            name = config.List_PumpQueueName % (self.project_id, client_id)

            # 开启线程处理发送数据到队列的任务
            thread = threading.Thread(target=self.send_data, args=(client_id, name))
            thread_list.append(thread)

        # 设置守护主进程
        for t in thread_list:
            t.setDaemon(True)
            t.start()
        t.join()


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


