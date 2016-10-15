#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 10:51:35 2016

@author: zxl
"""
from DataPretreatment import *


# 使用 logistic 回归进行训练和预测
def logistic_predict(data_train):
    clf = LogisticRegression()
    clf.fit(data_train[['price','Distance','Discount_rate']], data_train['label'])
    pre_proba = clf.predict_proba(data_train[['Discount_rate','Distance','Discount_rate']])
    pre_label = clf.predict(data_train[['Discount_rate','Distance','Discount_rate']])
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
    data_train['Discount_rate'], data_train['price'] = get_Discount_rate(data_train['Discount_rate'])
    data_train['Distance'] = distance_normal(data_train['Distance'])
    data_train['price'] = price_normal(data_train['price'])
    data_train['label'] = get_label(data_train['Date_received'], data_train['Date'])
#    data_train = get_sepSample(data_train)
#    tryfind(data_train)
    pre_proba, pre_label = logistic_predict(data_train)
#    pre_label = xgboost_predict(data_train)