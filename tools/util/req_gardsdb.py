import datetime
import numpy as np
import sys
import pandas as pd
import time
import ast

from util.http_util import HttpHelper

__DBAPI_URL = "http://af.gards.top:8888"
MOSAIC_URL = __DBAPI_URL + "/dbapi/mosaic"

http_helper = HttpHelper()


def dbapi_mosaic(client_id, duration_time, FIDIS_NAME, FIDIS_PWD):
    """
    :param client_id: 设备标识
    :param keys: 参数标识字符串，中间以逗号隔开， 如“参数标识1，参数标识2...”
    :param duration_time: 取值的时间时长, 单位：秒
    :param FIDIS_NAME: 用户名
    :param FIDIS_PWD: 用户密码
    :return:
    """
    req_url = MOSAIC_URL

    # 获取当前时间
    end_time = datetime.datetime.now()
    my_time = datetime.timedelta(seconds=5)
    end_time -= my_time
    create2 = end_time.strftime('%Y-%m-%d %H:%M:%S')

    # 获取抓取数据开始时间
    long_time = datetime.timedelta(seconds=duration_time)
    start_time = end_time - long_time

    create1 = start_time.strftime('%Y-%m-%d %H:%M:%S')
    # print(">>client_id={0},start_time={1},end_time={2}".format(client_id, create1, create2))

    data_params = {
        # 设备标识
        "client": client_id,
        # 获取数据开始时间
        "create1": create1,
        # 获取数据结束时间
        "create2": create2,
        # 时间间隔， 即获取数据的频率
        "interval": "5",
        # 时间单位， 可选值为 s，m，h。分别代表秒，分和小时
        "unit": "s",
        # 获取的数据类型
        "message_type": "r",
        # 参数标识字符串，中间以逗号隔开， 如“参数标识1，参数标识2...”
        # fangkuai
        "keys": "L1_3_4711,L1_3_4712,L1_3_4713,L1_3_4714,L1_3_4715,L1_3_4716,L1_3_4717,L1_3_4718,L1_3_4719,L1_3_4721,L1_3_4734,L1_3_4735,L1_3_4736,L1_3_4759,L1_3_4760,L1_3_4761,L1_3_4762,L1_3_4818",
        # jindong

        # jichai

        # comps

    }

    flag, content = http_helper.getdata(req_url, data_params, auth_username=FIDIS_NAME, auth_password=FIDIS_PWD)
    return flag, content


if __name__ == '__main__':

    # if len(sys.argv) != 5:
    #     print('Usage: {} clientid duration_time FIDIS_NAME FIDIS_PWD'.format(sys.argv[0]), file=sys.stderr)
    #     raise SystemExit(1)
    #
    # a = sys.argv[0]
    # clientid = sys.argv[1]
    # duration_time = int(sys.argv[2])
    # FIDIS_NAME = sys.argv[3]
    # FIDIS_PWD = sys.argv[4]

    # clientid="A22017051300771"
    # clientid="A22017051300883"
    # FIDIS_NAME = "jichai"
    # FIDIS_PWD = "jichai123"
    #
    # clientid="A22016122100154"
    # FIDIS_NAME = "jindong"
    # FIDIS_PWD = "jindong123"

    # clientid = "A22016122100140"
    clientid = "A22016122100271" #正常
    # clientid = "A22017051300120"  # 不正常
    # clientid = "A22017051300226"
    FIDIS_NAME = "fangkuai"
    FIDIS_PWD = "fangkuai123"

    # clientid = "A22017051300858" #空压机249
    # clientid = "A22017051300612" #电表60
    # clientid = "A22017051300865" #流量计60
    # FIDIS_NAME = "comps"
    # FIDIS_PWD = "comps123"

    duration_time = 60 * 60 * 24

    flag, content = dbapi_mosaic(clientid, duration_time, FIDIS_NAME, FIDIS_PWD)
    content = content.replace("null", "0")
    data = ast.literal_eval(content)
    param_name_list = data["key_list"]
    old_param_value_list = data["values_lists"]

    new_param_value_list = list()
    for i in old_param_value_list:
        new_param_value_list.append(i)

    my_list = list()
    for i in range(len(param_name_list)):
        value_list = list()
        for j in new_param_value_list:
            result = j[i]
            value_list.append(result)
        my_list.append(value_list)

    i_list = []
    j_list = []
    for i, j in zip(param_name_list, my_list):
        if i.startswith("L"):
            i_list.append(i)
            j_list.append(j)
            # j_list.append([float(k) for k in j])

    data_dict = dict()
    new_dict = dict()
    for i, j in zip(i_list, j_list):
        #############################
        # j = [float(k) for k in j]
        #############################
        new_dict[i] = j
        my_dict = dict()

        # 最大值
        param_max = max(j)
        # 最小值
        param_min = min(j)
        # 平均值
        param_average = float(sum(j)) / float(len(j))
        # 方差
        j_array = np.array(j)
        variance = np.var(j_array)

        my_dict["max"] = param_max
        my_dict["min"] = param_min
        my_dict["average"] = param_average
        my_dict["variance"] = variance
        # my_dict["values"] = j
        my_dict["len"] = len(j)
        my_dict["LargeMax"] = param_max * 2 - param_average
        my_dict["LargeMin"] = param_min * 2 - param_average

        data_dict[i] = my_dict

    df1 = pd.DataFrame(data_dict).T
    # csv文件储存路径
    # csvpath1 = "./A%s.csv" % time.strftime("%Y-%m-%d_%H:%M:%S")
    csvpath1 = "property.csv"
    df1.to_csv(csvpath1)

    df2 = pd.DataFrame(new_dict).T
    # csv文件储存路径
    # csvpath2 = "./B%s.csv" % time.strftime("%Y-%m-%d_%H:%M:%S")
    csvpath2 = "data.csv"
    df2.to_csv(csvpath2)
