import datetime

from util.http_util import HttpHelper

__DBAPI_URL = "http://af.gards.top:8888"
MOSAIC_URL = __DBAPI_URL + "/dbapi/mosaic"

# clientid="862631039219935"
# FIDIS_NAME = "xinlei"
# FIDIS_PWD = "xinlei123"

http_helper = HttpHelper()


def dbapi_mosaic(client_id, keys, duration_time, FIDIS_NAME, FIDIS_PWD):
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
    my_time = datetime.timedelta(seconds=60 * 5)
    end_time -= my_time
    create2 = end_time.strftime('%Y-%m-%d %H:%M:%S')

    # 获取抓取数据开始时间
    long_time = datetime.timedelta(seconds=duration_time)
    start_time = end_time - long_time

    create1 = start_time.strftime('%Y-%m-%d %H:%M:%S')
    print(">>client_id={0},start_time={1},end_time={2}".format(client_id, create1, create2))

    data_params = {

        # 设备标识
        "client": client_id,
        # 获取数据开始时间
        "create1": create1,
        # 获取数据结束时间
        "create2": create2,
        # 时间间隔， 即获取数据的频率
        "interval": "15",
        # 时间单位， 可选值为 s，m，h。分别代表秒，分和小时
        "unit": "s",
        # 获取的数据类型
        "message_type": "r",
        # 参数标识字符串，中间以逗号隔开， 如“参数标识1，参数标识2...”
        "keys": keys,

    }

    flag, content = http_helper.getdata(req_url, data_params, auth_username=FIDIS_NAME, auth_password=FIDIS_PWD)
    return flag, content
