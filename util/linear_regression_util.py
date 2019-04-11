import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_dataset(dataset, look_back):

    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


#####当激活函数为 sigmoid 或者 tanh 时，要把数据正则话，此时 LSTM 比较敏感 设定 99.2% 是训练数据，余下的是测试数据
def create_train_test_data(dataset,look_back, num):

    ############正则化
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    #print('dataset', dataset)
    trainX, trainY = create_dataset(dataset, look_back)

    return scaler, trainX, trainY

def predict_next(model, scaler, dataset, look_back, next_num):
    dataset_pre = scaler.transform([dataset[len(dataset) - look_back:len(dataset), 0]])

    # next_predicted_list = []
    # for i in range(next_num):
    #     dataXNext = dataset_pre[:, dataset_pre.shape[1] - look_back:]
    #     #dataPreNext = dataXNext.reshape(dataXNext.shape[0], look_back, 1)
    #     next_predicted = model.predict(dataXNext)
    #     next_predicted_array = np.array(next_predicted[0])
    #
    #     next_predicted_list.append(next_predicted)
    #
    #     dataset_pre = np.concatenate((dataset_pre, next_predicted_array.reshape(1, -1)), axis=1)
    #
    #     print("未来", i, "时刻的参数预测值为：", scaler.inverse_transform(next_predicted[0])[0][0])
    #
    # return scaler.inverse_transform(np.array(next_predicted_list).reshape(-1, len(next_predicted_list)))
    dataset_util = scaler.transform([dataset[len(dataset) - next_num:len(dataset), 0]])
    next_predicted_list = []
    for i in range(next_num):
        dataXNext = dataset_pre[:, dataset_pre.shape[1] - look_back:]
        #dataPreNext = dataXNext.reshape(dataXNext.shape[0], look_back, 1)
        next_predicted = model.predict(dataXNext)
        next_predicted_array = np.array([np.array(next_predicted[0])])

        next_predicted_list.append(next_predicted)
        new_final = dataset_util[0, next_num - i - 1]
        next_predicted_array[-1] = new_final

        dataset_pre = np.concatenate((dataset_pre, next_predicted_array.reshape(1, -1)), axis=1)

    return scaler.inverse_transform(next_predicted_list)

def line_regression_run(seq_len, future_predict_num, dataset):

    scaler, trainX, trainY = create_train_test_data(dataset, seq_len, future_predict_num)

    #建模
    model = LinearRegression(fit_intercept=True, n_jobs=1, normalize=False)
    model.fit(trainX, trainY)
    #模型训练
    testPredict = model.predict(trainX)
    testScore = metrics.mean_squared_error(trainY, testPredict)

    #预测未来数据
    next_predicted_list = predict_next(model, scaler, dataset, seq_len, future_predict_num)

    return next_predicted_list, testScore

