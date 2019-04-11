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
# Author :XinJinquan
# Description : filter过滤器

import json
import threading
import config
import sys
from daemon import Daemon
import filterScript

program_type = config.ProgramEnum.filter.name


class FilterDaemon(Daemon):
    def __init__(self, project=None):
        Daemon.__init__(self, project, program_type)
        self.project_id = None
        self.object_list = None
        self.data_filter = None
        self.data_organizer = None
        self.para_reference_name_list = None

    # 获取indass_bossf队列数据
    def set_project_info(self, project_info):
        try:
            # 获取project_id
            self.project_id = project_info["project_id"]

            # 获取object组
            self.object_list = project_info["object"]

            # 获取过滤器脚本
            self.data_filter = project_info["data_filter"]["script"]

            # 获取参数标识列表
            # self.para_reference_name_list = project_info["parameter"]

        except Exception as e:
            print("Failed to get the project info!：%s" % e)

    def script_parse(self, pump_data):
        """
        # TODO filter script parse
        :return:
            """
        filter_data = dict()
        for key in self.data_filter:
            if key[0] not in pump_data:

                pump_data[key[0]]= 0
            result = pump_data[key[0]]
            if result is None:
               return 0
            pump_data_value = result  # 取到pump的key
            data = str(key[1]).strip('@')  # 截取脚本中func关键词

            filter_script = getattr(filterScript, data)  # 获取脚本func
            try:
                script = filter_script(pump_data_value, key[2])  # 判断脚本文是否存在
                # 脚本是否满足条件！！！
                if script is True:
                    return 0  # 删除不符合脚本的元素
            except Exception as error:
                print("Script parsing error!", error)
            filter_data['time'] = pump_data['time']
            filter_data.update({key[0]: pump_data_value})  # 更新filter_data 字典
        return filter_data

    def get_object_list(self, client_id):
        while True:
            pump_queue_name = config.List_PumpQueueName % (self.project_id, client_id)
            res = config.redis_conn.brpop(pump_queue_name)[1]
            if not res:
                print('Get Project Info Error!')
                continue

            res = res.decode("utf-8")
            pump_data = json.loads(res)  # 转换成dict格式

            filter_data = self.script_parse(pump_data)
            if filter_data == 0:
                continue
            ####! 考虑字符情况
            for key in filter_data.keys():
                if filter_data[key] == '' or "":
                    filter_data[key] = 0

            if filter_data.values() is '' or filter_data.values is None or type(filter_data.values()) is str:
                del filter_data
                break
            filter_output_data = json.dumps(filter_data)  # 数据格式化
            output_name = config.List_FilterQueueName % (self.project_id,
                                                         client_id)  # 数据包命名
            if config.redis_conn.llen(output_name) > 301:
                config.redis_conn.rpop(output_name)

            print('filter output data is ->>>>>', filter_data)
            config.redis_conn.lpush(output_name, filter_output_data)


    def run(self):
        thread_list = []
        if self.object_list is not None:
            for object_dict in self.object_list:
                client_id = str(object_dict["object_id"])
                thread_list.append(threading.Thread(target=self.get_object_list, args=(client_id,)))
            try:
                for filter_thread in thread_list:
                    filter_thread.setDaemon(True)
                    filter_thread.start()
                filter_thread.join()

            except Exception as e:
                print("Failed to get the project info!：%s" % e)  ###############################################################################


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

            filter_daemon = FilterDaemon(project=project_id)
            filter_daemon.set_project_info(project_info)
            filter_daemon.start()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} [start|stop]'.format(sys.argv[0]), file=sys.stderr)
        raise SystemExit(1)

    queue_id = sys.argv[1]
    main(queue_id)
