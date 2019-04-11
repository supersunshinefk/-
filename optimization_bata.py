import time
import re

import sys
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import data_handle as dh
import config
import json
from daemon import Daemon
import ast
import time
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

program_type = config.ProgramEnum.optimization.name


class OptimizationDaemon(Daemon):

    def __init__(self, project):
        Daemon.__init__(self, project, program_type)
        self.script = None
        self.project_id = None
        self.para_list = None
        self.parm_list = None
        self.data_organizer = None
        self.optimal_parameter_list = None
        self.index_lis = None
        self.opt_para_quota_min_list = None
        self.opt_para_quota_max_list = None
        self.para_expect_list = None
        self.opt_para_reference_name_list = None
        self.parm_cn_list = None
        self.optimal_script = None

    #############################################################################

    def set_project_info(self, project_info):

        try:
            self.script = project_info["data_organizer"]["script"]["optimization_parameter"]
            self.project_id = project_info["project_id"]
            # 获取object组
            self.object_list = project_info["object"]
            # self.para_list = project_info["parameter"]
            # 获取组织器脚本
            self.data_organizer = project_info["data_organizer"]["script"]
            # 获取优化参数对象列表
            # self.optimal_parameter_list = project_info["optimal_parameter"]
            # self.optimal_script = project_info['optimal_parameter'][0]["script"]

            self.data_organizer_script = project_info["data_organizer"]["script"]
            self.organizer_analysis_parameter_list = self.data_organizer_script["analysis_parameter"]
            self.organizer_optimization_parameter_list = self.data_organizer_script["optimization_parameter"]

            self.analysis_parameter_list = project_info["analysis_parameter"]
            self.optimization_parameter_list = project_info['optimization_parameter']

            min_ = []  # 最小值
            max_ = []  # 最大值
            self.opt_para_quota_min_max_list = []
            self.min_max = []  # 最小值和最大值列表
            for list_ in self.optimization_parameter_list:
                min_.append(list_['para_quota_min'])

                max_.append(list_['para_quota_max'])
            self.opt_para_quota_min_max_list.append(min_)
            self.opt_para_quota_min_max_list.append(max_)


        except Exception as e:
            print("获取设备数据出错：%s" % e)

    def plan_one(self, a_list, PCA_list, opt_para_quota_min_list, opt_para_quota_max_list, opt_para_expect_list):
        a_list = a_list[1:]
        for i, j in zip(PCA_list[0], PCA_list[1]):
            my_list = []
            for index in j:
                my_list.append(a_list[index])
            my_array = np.array(my_list)
            param_list = my_array.T.tolist()

            coefficient_list, param_list0 = self.regression(a_list[i], param_list)

            # 参数个数
            param_count = len(param_list0)
            # 步长数组
            e_array = np.diag(param_list0) / 10
            # 定义一个赋值为0的50*参数个数的矩阵
            XX = np.zeros((10, param_count))
            YY = np.zeros((10, 1))

            coefficient_array = np.array(coefficient_list)
            end_array = np.array(param_list0)
            XX[0] = np.round(end_array)
            YY[0] = np.round(self.func(end_array, coefficient_array), 2)
            const = 1
            for k in range(9):
                yy = XX[k]
                y = np.array(yy[:])
                for l in range(param_count):
                    if self.feasible_region(y + const * e_array[l], opt_para_quota_min_list, opt_para_quota_max_list):

                        if (opt_para_expect_list[0] - self.func(y + const * e_array[l], coefficient_array)) ** 2 <= (
                                opt_para_expect_list[0] - self.func(y, coefficient_array)) ** 2:
                            y += const * e_array[l]

                    if self.feasible_region(y - const * e_array[l], opt_para_quota_min_list, opt_para_quota_max_list):
                        if (opt_para_expect_list[0] - self.func(y - const * e_array[l], coefficient_array)) ** 2 <= (
                                opt_para_expect_list[0] - self.func(y, coefficient_array)) ** 2:
                            y -= const * e_array[l]
                if self.func(XX[k + 1], coefficient_array) == self.func(XX[k], coefficient_array):
                    const *= 0.8
                if type(y) is np.ndarray:
                    XX[k + 1] = np.around(y, 2)
                else:
                    XX[k + 1] = round(y, 2)
                if type(self.func(XX[k + 1], coefficient_array)) is np.ndarray:

                    YY[k + 1] = np.around(self.func(XX[k + 1], coefficient_array), 2)
                else:

                    YY[k + 1] = round(self.func(XX[k + 1], coefficient_array), 2)
            optimization_x = XX.T.tolist()
            optimization_y = YY.T
            if type(optimization_y) is np.ndarray:
                optimization_y = (np.around(optimization_y, 2)).tolist()
            else:
                optimization_y = (np.around(optimization_y, 2)).tolist()
            return optimization_x, optimization_y

    def plan_two(self, a_list, PCA_list, opt_para_quota_min_list, opt_para_quota_max_list, opt_para_expect_list):
        PCA_list1=PCA_list
        a_list = a_list[1:]

        y_data=a_list[0]
        param_list=a_list[1:]

        WeightsAndBiases, param_list0 = self.BPregression(y_data, param_list)

        # 参数个数
        param_count = len(param_list0)
        # 步长数组
        e_array = np.diag(param_list0) / 5
        # 定义一个赋值为0的50*参数个数的矩阵
        XX = np.zeros((10, param_count))
        YY = np.zeros((10, 1))

        end_array = np.array(param_list0)
        XX[0] = end_array
        YY[0] = self.BPfunc(end_array, WeightsAndBiases)
        const = 1

        for k in range(9):
            yy = XX[k]
            y = np.array(yy[:])
            for l in range(param_count):
                if self.feasible_region(y + const * e_array[l], opt_para_quota_min_list, opt_para_quota_max_list):

                    if (opt_para_expect_list[0] - self.BPfunc(y + const * e_array[l], WeightsAndBiases)) ** 2 <= (
                            opt_para_expect_list[0] - self.BPfunc(y, WeightsAndBiases)) ** 2:
                        y += const * e_array[l]

                if self.feasible_region(y - const * e_array[l], opt_para_quota_min_list, opt_para_quota_max_list):
                    if (opt_para_expect_list[0] - self.BPfunc(y - const * e_array[l], WeightsAndBiases)) ** 2 <= (
                            opt_para_expect_list[0] - self.BPfunc(y, WeightsAndBiases)) ** 2:
                        y -= const * e_array[l]
            if self.BPfunc(XX[k + 1], WeightsAndBiases) == self.BPfunc(XX[k], WeightsAndBiases):
                const *= 0.8

            XX[k+1]=y
            YY[k+1]=self.BPfunc(XX[k + 1], WeightsAndBiases)

        optimization_x = np.round(XX,2).T.tolist()
        optimization_y = np.round(YY,2).T.tolist()

        return optimization_x, optimization_y


    def process_optimization(self, sub_organizer_channel):
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

            organizer_queue_name = config.Hset_Organizer_Optimization_QueueName.format(
                project_id=project_id, object_id=object_id)
            # if config.redis_conn.hlen(organizer_queue_name) >= config.Max_Dispenser_Queue_Size:

            print(">>>开始时间", time.time())
            while True:

                key_list = config.redis_conn.hkeys(organizer_queue_name)
                if key_list:
                    break
            max_key = max(key_list)
            while True:
                data = config.redis_conn.hget(organizer_queue_name, max_key)
                if data:
                    break
                else:
                    print("optimization 获取数据为空！！！")
                    time.sleep(5)
            data = json.loads(data.decode('utf-8'))
            value_matrix = np.mat(data)
            # origin_data = pd.DataFrame(value_matrix)
            # symbol, index_list = dh.analytical_symbol(self.optimal_script, self.parm_list)
            # value_matrix = np.mat(dh.organize_dataset(origin_data, symbol, index_list))
            new_list = self.choose_parameter(value_matrix, self.index_list)
            PCA_list = self.principal_component_analysis(value_matrix, new_list)

            #     '这是一些参数 --->>>> %s参数1  %s 参数2 %s 参数3 %s 参数4 %s 参数5 ' % (
            #     param_list, PCA_list, self.opt_para_quota_min_list, self.opt_para_quota_max_list,
            #     self.opt_para_expect_list))
            new_opt_para_quota_min_list = []
            new_opt_para_quota_max_list = []
            print('pca list ->>>', PCA_list)
            for i in PCA_list[1][0]:
                new_opt_para_quota_min_list.append(self.opt_para_quota_min_max_list[0][i])
                new_opt_para_quota_max_list.append(self.opt_para_quota_min_max_list[1][i])

            new_opt_para_expect_list = []
            for i in PCA_list[0]:
                new_opt_para_expect_list.append(self.para_expect_list[i])

            plan_one_data = self.plan_one(data, PCA_list, new_opt_para_quota_min_list,
                                          new_opt_para_quota_max_list, new_opt_para_expect_list)

            plan_two_data = self.plan_two(data, PCA_list, self.opt_para_quota_min_max_list[0],
                                          self.opt_para_quota_min_max_list[1], self.para_expect_list)

            para_variate = dict(map(lambda x, y: [x, y], self.parm_list, self.parm_cn_list))
            my_index1 = PCA_list[0][0]
            my_index2 = PCA_list[1][0]
            param_name_list = []
            for i in my_index2:
                param_name_list.append(para_variate[self.new_parameter_name_list[i]])

            variate = self.new_parameter_name_list[my_index1]

            param_name1 = para_variate[variate]

            plan_one_dict = dict()

            plan_one_dict["Line_path"] = {param_name1: plan_one_data[1][0]}

            param_dict = dict()
            for i, j in zip(param_name_list, plan_one_data[0]):
                param_dict[i] = j

            plan_one_dict["Parameter"] = param_dict
            # print('optimization output -111111111111111', plan_one_dict)

            plan_two_dict = dict()

            plan_two_dict["Line_path"] = {param_name1: plan_two_data[1][0]}

            # my_list = []
            # for i in plan_two_data:
            #     print('plan two data is ---->', plan_two_data)
            #     my_list.append(i[:])
            param_dict2 = dict()
            for k, l in zip(param_name_list, plan_two_data[0]):
                param_dict2[k] = l

            plan_two_dict["Parameter"] = param_dict2

            print('optimization output ->>>>>>>>>>>>>@@@', plan_one_dict)

            if plan_one_dict:
                plan_one_json = json.dumps(plan_one_dict)

                plan_one_queue_name = config.HSet_Opti_One_QueueName.format(project_id=project_id,
                                                                            object_id=object_id)
                config.redis_conn.hset(plan_one_queue_name, str(data_block_id), plan_one_json)
            if plan_two_dict:
                plan_two_json = json.dumps(plan_two_dict)
                plan_tow_queue_name = config.HSet_Opti_Two_QueueName.format(project_id=project_id,
                                                                            object_id=object_id)
                # config.redis_conn.hset(plan_tow_queue_name, str(data_block_id), plan_two_json)


            # 判断redis队列中长度是否超过Max_Compute_Queue_Size条，超过以后,进行删除最旧的一条计算数据
            if config.redis_conn.hlen(plan_one_queue_name) > config.Max_Compute_Queue_Size:
                key_list = config.redis_conn.hkeys(plan_one_queue_name)
                new_key_list = []
                for keys in key_list:

                    key_min = keys.decode('utf-8')
                    new_key_list.append(int(key_min))
                key = min(new_key_list)
                config.redis_conn.hdel(plan_one_queue_name, key)
            else:
                pass
            print(">>>结束时间", time.time())

    #############################################################################

    def choose_parameter(self, value_matrix, initial_parameter):
        """
        :param value_matrix: 所有参数各时刻值
        :param initial_parameter: 所选参数对应的行数组成的列表
        :return:
        """
        # delete value_matrix's time of the number one line
        value_matrix = np.delete(value_matrix, 0, axis=0)
        # 参数个数
        rows_value = value_matrix.shape[0]
        # 所选参数个数
        initial_number = len(initial_parameter)
        # 所选参数偏度列表
        skew_list = np.zeros(initial_number)
        # 所选参数峰度列表
        kurt_list = np.zeros(initial_number)
        skew_a = np.zeros(initial_number)

        for i in range(initial_number):
            skew_a[i] = value_matrix[initial_parameter[i]][0, i]
            skew_list[i] = np.sum(((skew_a - np.mean(skew_a)) / (
                    np.var(skew_a) ** 0.5)) ** 3 / len(skew_a))
            kurt_list[i] = np.sum(((skew_a - np.mean(skew_a)) / (
                    np.var(skew_a) ** 0.5)) ** 4 / len(skew_a))
        para_list = []
        skew_list = skew_list.tolist()
        # kurt_list = kurt_list.tolist()
        for k in range(initial_number):
            # 筛选出偏度大于一，峰度小于3的参数
            skew_list[k] = abs(skew_list[k])
            #   if kurt_list[k] < 3:
            para_list.append(initial_parameter[k])
        # 如果列表为空，即无满足可优化条件的参数，则选择偏度最大的参数进行优化
        if len(para_list) == 0:
            only_choice = initial_parameter[skew_list.index(max(skew_list))]
            para_list.append(only_choice)
            # para_list.append(initial_parameter[only_choice])
        para_list = [para_list[skew_list.index(max(skew_list))]]
        feasible_number = len(para_list)
        able_corr_list = []

        for w in range(feasible_number):
            # 如果此参数方差为零则不做优化
            # if np.cov(value_matrix[para_list[w]]) == 0:
            #
            #
            #
            #     continue
            corr_list = []
            original_corr_list = []
            original_corr_list_all = []
            # 对方差不为零的参数进行与其他参数的相关系数计算，挑选出系数大于0.6的
            for e in range(rows_value):
                Corr_WE = (np.corrcoef(value_matrix[para_list[w]], value_matrix[e]))[0, 1]
                Corr_WE = np.nan_to_num(Corr_WE)
                original_corr_list_all.append(Corr_WE)
                if (Corr_WE > 0.4) & (para_list[w] != e):
                    corr_list.append(e)
                if (para_list[w] != e):
                    original_corr_list.append(Corr_WE)
            max_corr = original_corr_list_all.index(max(original_corr_list))
            # 对于所要优化的参数，满足以上条件的参数组成一个列表
            if len(corr_list) != 0:
                able_corr_list.append(corr_list)
            # 如果所有需要优化的参数都没有其他参数与其相关系数超过0.6。则没有可优化的参数
            if len(able_corr_list) == 0:
                able_corr_list.append([max_corr])
        new_list = [para_list, able_corr_list]
        return new_list

    #############################################################################

    def principal_component_analysis(self, value_matrix, parameter_tuple):
        value_matrix = np.delete(value_matrix, 0, axis=0)
        PCA_list = []
        PCA_list.append(parameter_tuple[0])
        relevant_param_list = []
        for parameter_list in parameter_tuple[1]:

            if len(parameter_list) <= 4:
                relevant_param_list.append(parameter_list)
                continue

            pca_compute_list = []

            for i in parameter_list:
                pca_compute_list.append(np.array(value_matrix[i])[0])
            # 需要计算的pca矩阵
            pca_compute_matrix = np.array(pca_compute_list)
            # 创建一个字典，用于数据还原的对应序列的关系 x1,x2,x3,,,,（相当于论文中的表7）
            pca_restore = {}

            pca = PCA(n_components=0.80)  # PCA方法参数n_components
            # 如果将其设置为小数，则说明降维后的数据能保留的信息。
            pca.fit(pca_compute_matrix.T)  # .T转置
            variance = pca.explained_variance_ratio_
            # explained_variance_ratio_
            # print(variance) #方差贡献率
            meanVals = np.mean(pca_compute_matrix, axis=0)  # 均值
            meanRemoved = pca_compute_matrix - meanVals

            covMat = np.cov(meanRemoved)  # 协方差

            eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 计算特征值、特征向量

            eigValInd = np.argsort(eigVals)  # 从大到小排序，返回索引值数组

            feaeigValInd = eigValInd[:-(len(variance) + 1):-1]  # 根据方差的贡献率来看保留几个值（主成分）保留后topNfeat个,然后调转先后

            redEigVects = eigVects[:, feaeigValInd]  # 按feaeigValInd排序和去掉列
            # 降维特征向量，也叫降维后的10*4的成分矩阵
            # 新的特征向量，新的成分矩阵，新的载荷矩阵
            # eigVals_new = eigVals[0:len(variance)]  # 选出前len(variance)中，其为最大的特征值
            eigVals_new = []
            for i in feaeigValInd:
                eigVals_new.append(eigVals[i])
            # 表5、表6的计算 降维后的特征值
            for i in range(len(eigVals_new)):
                redEigVects[:, i] = redEigVects[:, i] / np.sqrt(eigVals_new[i])
                redEigVects[:, i] = redEigVects[:, i] * eigVals_new[i]

            redEigVects_new = np.dot(redEigVects, variance.T)  # 相乘,得到表7(论文)

            # 将（论文中的表7）加绝对值
            redEigVects_new_fabs = np.abs(redEigVects_new).tolist()

            # pca_restore = dict(map(lambda x,y:[x,y],parameter,redEigVects_new_fabs[0]))
            pca_restore = dict(map(lambda x, y: [x, y], parameter_list, redEigVects_new_fabs[0]))
            # 将字典中的values（相当于论文中的表7）进行大小排序
            pca_restore_sort = sorted(pca_restore.items(), key=lambda x: x[1], reverse=True)

            pca_finish = pca_restore_sort[0:4]

            list_ = []  # 降维的列表参数下标
            for i in pca_finish:
                list_.append(i[0])

            relevant_param_list.append(list_)
        PCA_list.append(relevant_param_list)
        return PCA_list

    #############################################################################

    def feasible_region(self, param_list0, opt_para_quota_min_list, opt_para_quota_max_list):
        """
        :param param_list: 参数列表
        :param opt_para_quota_min_list: 最小阈值列表
        :param opt_para_quota_max_list: 最大阈值列表
        :return: 0或1
        """
        for para, min, max in zip(param_list0, opt_para_quota_min_list, opt_para_quota_max_list):
            if (para >= min).all() and (para <= max).all():
                pass
            else:
                return 0
        return 1

    def func(self, matrix, coefficient_array):
        """
        :param matrix: [[x1, x2, x3...]]
        :param coefficient_array:
        :return:
        """
        return np.dot(matrix, coefficient_array[1:]) + coefficient_array[0]

    def regression(self, obj, param_list):
        """
        :param obj: y的300个数列表
        :param param_list:
        :return:
        """

        model = LinearRegression()
        model.fit(param_list, obj)

        # 常数项系数
        a0 = model.intercept_
        # 各项系数集合
        a = model.coef_

        # 系数列表
        coefficient_list = []
        coefficient_list.append(a0)
        for i in a:
            coefficient_list.append(i)

        return coefficient_list, param_list[-1]

    #############################################################################

    def BPregression(self, y_data, x_data):
        # x_data = np.array([[1, 1], [2, 2], [1, 2], [1, -1], [2, 0], [2, -1]] * 2000, dtype=np.float32)
        # x_test = np.array([[2, -1], [2, 0], [1, -1], [1, 2], [2, 2], [1, 1]], dtype=np.float32)
        # y_data = np.array([[1], [2], [3], [4], [5], [6]] * 2000, dtype=np.float32)
        x_data=np.array(np.array(x_data).T.tolist())
        y_data=np.array(np.array([y_data]).T.tolist())
        input_num = x_data.shape[1]
        output_num = y_data.shape[1]
        hide_num = (input_num + output_num) * 2

        x = tf.placeholder(tf.float32, shape=(None, input_num))  # None随batch大小变化
        y = tf.placeholder(tf.float32, shape=(None, output_num))

        w1 = tf.Variable(tf.random_normal([input_num, hide_num], stddev=1, seed=1))
        w2 = tf.Variable(tf.random_normal([hide_num, output_num], stddev=1, seed=1))
        b1 = tf.Variable(tf.zeros([hide_num]))
        b2 = tf.Variable(tf.zeros([output_num]))

        # a = tf.nn.relu(tf.matmul(x,w1) + b1)
        a = tf.nn.tanh(tf.matmul(x, w1) + b1)
        y_re = tf.matmul(a, w2) + b2

        # cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_, name=None)
        cost = tf.reduce_mean(tf.square(y_data - y_re))
        # train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
        train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for i in range(300):
                sess.run(train_op, feed_dict={x: x_data, y: y_data})
            weight1 = sess.run(w1)
            biase1 = sess.run(b1)
            weight2 = sess.run(w2)
            biase2 = sess.run(b2)
            # print(type(weight1), type(biase1), type(weight2), type(biase2))
        WeightsAndBiases = [weight1.tolist(), biase1.tolist(), weight2.tolist(), biase2.tolist()]
        return WeightsAndBiases,x_data[-1]

    def BPfunc(self, x_test, WeightsAndBiases):
        weight1 = np.array(WeightsAndBiases[0])
        biase1 = np.array(WeightsAndBiases[1])
        weight2 = np.array(WeightsAndBiases[2])
        biase2 = np.array(WeightsAndBiases[3])
        A = np.tanh(np.matmul(x_test, weight1) + biase1)
        Results = np.matmul(A, weight2) + biase2
        return Results

    #############################################################################
    def run(self):
        # parameter_name_list = re.findall(config.regex, self.script)
        #
        # para_reference_name_obj = re.findall(r'\n?.+=', self.script)
        # para_reference_name_list = []
        # self.parm_cn_list = []
        #
        self.parm_list = []
        self.parm_cn_list = []
        for parm in self.optimization_parameter_list:
            self.parm_list.append(parm['para_id'])
            self.parm_cn_list.append(parm['para_name_cn'])

        # for obj in para_reference_name_obj:
        #     para_reference_name_list.append(obj.strip("=\n"))
        self.para_expect_list = []
        # for para_data in self.para_list:
        for i in self.optimization_parameter_list:
            self.para_expect_list.append(i["para_expect"])
            self.para_expect_list.append(i["para_quota_max"])
            self.para_expect_list.append(i["para_quota_min"])
        # "para_reference_name": "L1_3_4"

        # parameter_list = self.analysis_parameter_list + self.optimization_parameter_list
        #
        # parameter_list = list(set(parameter_list))
        # for obj in parameter_list:
        #     self.para_expect_list.append(obj["para_expect"])
        #     self.opt_para_quota_max_list.append(obj["para_quota_max"])
        #     self.opt_para_quota_min_list.append(obj["para_quota_min"])
        #     self.parm_cn_list.append(obj["para_name"])

        # 获取要优化参数的下标
        self.index_list = [0]

        new_parameter_name_list = []
        # for parameter_name in parameter_name_list:
        #     para = parameter_name[1:]
        #     new_parameter_name_list.append(para)
        for i in self.organizer_optimization_parameter_list:
            new_parameter_name_list.append(i[0])
        for i in self.organizer_optimization_parameter_list:
            if i[0] not in self.organizer_optimization_parameter_list:
                new_parameter_name_list.append(i[0])

        self.new_parameter_name_list = new_parameter_name_list
        sub_organizer_channel = []

        if self.object_list is not None:
            for object_dict in self.object_list:
                client_id = str(object_dict["object_id"])
                # thread_list.append(Thread(target=self.send_data, args=(new_parameter_name_list, client_id)))
                channel_name = config.Channel_OptimiizationName.format(project_id=self.project_id,
                                                                       object_id=client_id)
                sub_organizer_channel.append(channel_name)
            self.process_optimization(sub_organizer_channel)


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
            optimization_daemon = OptimizationDaemon(project=project_id)
            # print('info',project_info)
            optimization_daemon.set_project_info(project_info=project_info)
            optimization_daemon.start()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} [start|stop]'.format(sys.argv[0]), file=sys.stderr)
        raise SystemExit(1)
    queue_id = sys.argv[1]
    main(queue_id)
