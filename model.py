#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 10:51:35 2016

@author: zxl
"""

from DataPretreatment import *
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import KFold
import xgboost as xgb


# 使用 logistic 回归进行训练和预测
def logistic_predict(data_train):
    kf = KFold(data_train.shape[0], n_folds=10, shuffle=True)
    for train_index, test_index in kf:
        trainData = data_train.iloc[train_index]
        testData = data_train.iloc[test_index]
        clf = LogisticRegression()
        clf.fit(trainData[['price','Distance','Discount_rate']], trainData['label'])
        pre_proba = clf.predict_proba(testData[['Discount_rate','Distance','Discount_rate']])[:,1]
        pre_label = clf.predict(testData[['Discount_rate','Distance','Discount_rate']])
        test_auc = metrics.roc_auc_score(testData['label'], pre_proba)
        print test_auc
    return pre_proba, pre_label
    
    
# 使用 xgboost 进行训练和预测
def xgboost_predict(data_train):
    dtrain = xgb.DMatrix(data_train[['price','Distance']], label=data_train['label'])
    param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic', 'scale_pos_weight':0.006 }
    param['nthread'] = 4
    plst = param.items()
    plst += [('eval_metric', 'auc')]
    num_round = 10
    bst = xgb.train( plst, dtrain, num_round)
    pre_label = bst.predict(dtrain)
    return pre_label
    

if __name__ == '__main__':
    data_train, data_test = loadData()
    data_train = handleNA(data_train)
    data_train['Discount_rate'], data_train['price'] = get_Discount_rate(data_train['Discount_rate'])
#    data_train['Distance'] = distance_normal(data_train['Distance'])
#    data_train['price'] = price_normal(data_train['price'])
    data_train['label'] = get_label(data_train['Date_received'], data_train['Date'])
#    data_train = get_sepSample(data_train)
#    tryfind(data_train)
    pre_proba, pre_label = logistic_predict(data_train)
#    pre_label = xgboost_predict(data_train)