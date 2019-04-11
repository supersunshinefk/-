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
            print("获取indass_boss数据错误：%s" % e)

    # 获取历史数据
    def get_first_time_data(self, equipment_id, name):
        data = {
            "equipment_id": equipment_id,
            "items": config.APIX_By_items
        }
        try:
            response = requests.post(config.APIX_URL, data=data)
            receive_data = ast.literal_eval(response.text)
        except Exception as e:
            print("拿取apix历史数据错误,%s秒后重新获取数据" % config.TIME_INTERVAL)
            return 1
        code = receive_data["code"]
        if code != 200:
            print("获取fidis数据失败,%s秒后重新获取数据" % config.TIME_INTERVAL)
            return 1
        result_list = receive_data["result"]
        if len(result_list) == 0:
            print("获取fidis数据为空,%s秒后重新获取数据" % config.TIME_INTERVAL)
            return 1
        data_time_list = list()
        all_data_list = list()
        for result in result_list:
            value_dict = result["value"]
            all_data_list.append(value_dict)
            time_array = time.strptime(result["time"], "%Y-%m-%d %H:%M:%S")
            time_stamp = int(time.mktime(time_array))
            data_time_list.append(time_stamp)
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
        return 0

    # 获取数据，处理数据
    def get_other_time_data(self, equipment_id, now_time, name):
        data = {
            "equipment_id": equipment_id,
            "items": 1
        }
        try:
            response = requests.post(config.APIX_URL, data=data)
            receive_data = ast.literal_eval(response.text)
        except Exception as e:
            print("请求apix实时数据错误,%s秒后重新获取数据" % config.TIME_INTERVAL)
            return 1
        code = receive_data["code"]
        if code != 200:
            print("获取fidis数据失败,%s秒后重新获取数据" % config.TIME_INTERVAL)
            return 1
        result_list = receive_data["result"]
        if len(result_list) == 0:
            print("获取fidis数据为空,%s秒后重新获取数据" % config.TIME_INTERVAL)
            return 1
        for result in result_list[::-1]:
            result["value"]["time"] = now_time
            print(now_time)
            json_data = json.dumps(result["value"])
            print(json_data)
            # 把数据放到redis队列
            config.redis_conn.lpush(name, json_data)
        return 0

    # 定义发送数据到队列的任务
    def send_data(self, client_id, name):
        is_first = True
        num = 2
        while True:
            if is_first:
                print(">>第1次存储数据")
                result = self.get_first_time_data(client_id, name)
                if result == 1:
                    time.sleep(config.TIME_INTERVAL)
                    continue
                is_first = False
                time.sleep(config.TIME_INTERVAL)
                continue
            now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(">>第%d次存储数据" % num)
            start_time = time.time()
            result = self.get_other_time_data(client_id, now_time, name)
            end_time = time.time()
            elapsed_time = end_time-start_time
            # 设置时间延迟再次获取数据
            sleep_time = config.TIME_INTERVAL - elapsed_time
            if sleep_time < 0:
                print("拿取数据请求时间大于时间间隔")
                num += 1
                continue
            if result == 1:
                time.sleep(sleep_time)
                continue
            time.sleep(sleep_time)
            num += 1

    def run(self):
        # 定义一个列表存放线程
        thread_list = []
        while True:
            if not self.object_list:
                print("The analysis object is empty, Please check if there is an analysis object in redis")
                time.sleep(5)
                continue
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
            break


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
