# -*- coding: utf-8 -*- 

from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from rgf import RGFRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from datapreprocess import *

def readData(path, workMode):
    if workMode == 'train':
        label = []
        feature = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().replace("\t", '').split(" ")
                feature.append(line[1:])
                label.append(line[0])
        return np.array(feature).astype(np.float64), np.array(label).astype(np.float64)
    if workMode == 'test':
        feature = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().replace("\t", '').split(" ")
                feature.append(line)
        return np.array(feature).astype(np.float64)


def RGF_trainAndPredict():
    traindataPath = r'train_featureLength_3_hot.txt'
    testdataPath = r'test_featureLength_3_hot.txt'

    X, y = readData(traindataPath, 'train')
    X[np.isnan(X)] = -0.00001
    y[np.isnan(y)] = 0

    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    print(1)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(2)

    rgf = RGFRegressor(verbose=False)
    print(3)

    parameters = {
        'max_leaf': [2000, 4000, 6000, 8000],
        'algorithm': ['RGF_Sib'],
        'loss': ['LS', 'Log', 'Expo'],
        'test_interval': [100, 300, 500],
        'l2': [0.01, 0.1, 1.0],  # 随机采样训练样本
        'learning_rate': [0.03],
        #objective = 'reg:linear',
    }
    """
    parameters = {
            'max_leaf': [1000],
            'algorithm': ['RGF'],
            'loss': ['LS', 'Expo'],
            'test_interval': [300],
            'l2': [0.1], # 随机采样训练样本
            'learning_rate': [0.01],
            #objective = 'reg:linear',
    }
    """
    grid_search = GridSearchCV(
        rgf, parameters, scoring="neg_mean_absolute_error", cv=5)
    print(4)
    print(grid_search)
    #print (X_train, y_train,1),(len(X_train), len(y_train),1),np.max(X_train)
    grid_result = grid_search.fit(X_train, y_train)
    print(5)
    print(grid_search.grid_scores_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print("2")

    y_predict = grid_search.predict(X_test)

    test_error = mean_absolute_error(y_test, y_predict)

    print('the tset error is: ', test_error)
    print('the weighted tset error is: ',
          test_error * len(y_test) / np.sum(y_test))

    X_predict = readData(testdataPath, 'test')
    X_predict[np.isnan(X_predict)] = 0

    y_hat = grid_search.predict(X_predict)
    print("3")
    # resultPath=r'result.csv'
    featureLength = 3
    with open("predict_" + str(test_error) + '_' + str(featureLength) + "_new.csv", "w", encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        for i in range(0, 300):
            writer.writerow([i + 1, np.abs(y_hat[i])])


def main():
    #dataPreprocess()
    featureExtract_train(3)
    # featureExtract_test()
    # RGF_trainAndPredict()
    print 'task finished'


if __name__ == '__main__':
    main()
