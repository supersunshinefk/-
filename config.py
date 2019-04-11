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
# Author : Joy & XinJinquan
# Description : 配置文件



# -*- coding: utf-8 -*-

import redis
import pymysql
from influxdb import InfluxDBClient

# mysql
HOST = "127.0.0.1"
USER = 'root'
DB = 'indass_boss'
PASSWORD = 'passw0rd'
PORT = 3306
CHARSET = "UTF8"

# redis
REDIS_HOST = '192.168.88.17'
REDIS_PORT = 6378
REDIS_PWD = 'mixlinker@888'
REDIS_DB = 0

# influxdb
Influxdb_HOST = "192.168.88.17"
Influxdb_USER = "admin"
Influxdb_DB = "indass_history"
Influxdb_PASSWORD = "admin123"
Influxdb_PORT = 8087

###############################################################################
# 原数据输入输出队列List
List_PumpQueueName = 'list:PumpQ:%s:%s'
# 过滤器输入输出队列List
List_FilterQueueName = 'list:FilterQ:%s:%s'
# 组织器输入输出队列List
List_OrganizerQueueName = 'list:OrganizerQ:%s:%s'
# 组织器输出队列 hset
Hset_OrganizerQueueName = 'hset:OrganizerQ:{project_id}:{object_id}'
# 分发器输入输出队列HashSet
HSet_DispenserQueueName = 'hset:DispenserQ:%s:%s'
# 原始数据HashSet
Hset_OriginalQueueName = 'hset:OriginalQ:{project_id}:{object_id}'
##
## 定义输出队列，算法用
# map, index, stb, tendency 所需队列
Hset_Organizer_Analysis_QueueName = 'hset:Hset_Organizer_Analysis_Q:{project_id}:{object_id}'  # 算法队列
# 优化算法队列

### 数据输出队列
Hset_Organizer_Optimization_QueueName = 'hset:Hset_organizer_Optimzation_Q:{project_id}:{object_id}'
Incr_OrganizerName = 'incr:organzier:{project_id}:{object_id}:id'
Incr_OptimiizationName = 'incr:optimization:{project_id}:{object_id}:id'

# 订阅队列
Channel_OrganizerName = 'channel:OrganizerQ:{project_id}:{object_id}'
Channel_OptimiizationName = 'channel:optimization:{project_id}:{object_id}:id'

# Channel_DispenserName = 'channel:DispenserQ:{project_id}:{object_id}'
# 分发队列中保存的最大数据
Max_Organzier_Queue_Size = 1

# 判断算法计算的redis队列（包括combine）中保存的最大数据为10000条
Max_Compute_Queue_Size = 30

# 延时2s从Map,Index,Stab,Tend,Risk queue获取数据
Max_Combine_Interval = 2

# 合并输入输出队列
HSet_CombineOut_Queue = 'hset:CombineQ:{project_id}:{object_id}'
Channel_CombineOut_Queue = 'channel/{customer_id}/{project_id}/{object_id}'

###############################################################################
# 映象输入输出队列List
HSet_MapQueueName = 'hset:MapQ:{project_id}:{object_id}'
# 指数输入输出队列List
HSet_IndexQueueName = 'hset:IndexQ:{project_id}:{object_id}'
# 稳定性输入输出队列List
HSet_StabQueueName = 'hset:StabQ:{project_id}:{object_id}'

# 趋势输入输出队列List
HSet_TendQueueName = 'hset:TendQ:{project_id}:{object_id}'
# 风险输入输出队列List
HSet_RiskQueueName = 'hset:RiskQ:{project_id}:{object_id}'

# 优化输出队列 list
HSet_Opti_Two_QueueName = 'hset:Opti:Two:{project_id}:{object_id}'
HSet_Opti_One_QueueName = 'hset:Opti:One:{project_id}:{object_id}'

###############################################################################
LIST_BOSS_PROJECT = "list:boss:projects"
HSET_BOSS_PROJECT = "hset:boss:projects"

INTERPRETER = "/usr/local/bin/python3"
TEMP_PID_PATH = "/tmp/indass/"
PROGRAM = ["pump_by_apix.py", "filter.py", "organizer.py", "map.py", "index.py", "stability.py", "tendency.py",
           "risk.py", "optimization.py", "combine.py", "message.py"]

regex = r"[=: ?][A-Za-z]+\d+"
sched_delay = 5

from enum import Enum

class ProgramEnum(Enum):
    pump = 0
    filter = 1
    organizer = 2
    dispenser = 3
    map = 4
    index = 5
    stability = 6
    tendency = 7
    risk = 8
    optimization = 9
    combine = 10
    message = 11


###############################################################################

# message获取的连接地址
login_url = "http://192.168.1.242/fidis/fidis_admin/public/api/login"
# message发送连接
message_url_body = '192.168.1.242/fidis/fidis_admin/public/api/message/add'
url = "http://" + message_url_body

### organier need FilterQ len
llen = 300

## 原始数据队列长度
ORGANIZER_QUEUE_LINE_MAX = 5000

###############################################################################
# pump配置参数

# 拿取数据的时间间隔, 单位：秒
TIME_INTERVAL = 5
# 计算数据的条数
DATA_COUNT = 300
# 请求的马赛克数据条数
PAGE_SIZE = 1000
# apix的url地址
# APIX_URL = "http://rnd.mixiot.top:18090/v1/apix/mosaicByKey"
APIX_URL = "http://rnd.mixiot.top:18090/v1/apix/latestMosaicByItems"
APIX_By_items = 6000


###############################################################################

#Connected MYSQL
def conn():
    conn = pymysql.connect(host=HOST, user=USER, db=DB, passwd=PASSWORD,
                           port=PORT, charset=CHARSET)
    return conn

#Connected Redis
global redis_conn
redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,  db=REDIS_DB)

# Connected InfluxDB
influxdb_conn = InfluxDBClient(host=Influxdb_HOST, port=Influxdb_PORT, username=Influxdb_USER,password=Influxdb_PASSWORD,
                               database=Influxdb_DB)
