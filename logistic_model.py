#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 10:51:35 2016

@author: zxl
"""

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from DataBuffer import *
import xgboost as xgb


def logistic_tuning(data_train):
    param_grids = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    clf = LogisticRegression()
    clf = GridSearchCV(estimator=clf, param_grid=param_grids, scoring='roc_auc', n_jobs=5)
    clf.fit(data_train[['price','Distance']], data_train['label'])
    print clf.best_params_
    print clf.best_score_
    return clf.best_params_


# 使用 logistic 回归在训练集上进行交叉验证
def logistic_trainData_predict(data_train, best_params):
    kf = KFold(data_train.shape[0], n_folds=10, shuffle=True)
    for train_index, test_index in kf:
        trainData = data_train.iloc[train_index]
        testData = data_train.iloc[test_index]
        clf = LogisticRegression(C=best_params['C'])
        clf.fit(trainData[['price','Distance']], trainData['label'])
        pre_proba = clf.predict_proba(testData[['Discount_rate','Distance']])[:,1]
        pre_label = clf.predict(testData[['Discount_rate','Distance']])
        test_auc = metrics.roc_auc_score(testData['label'], pre_proba)
        print test_auc
    return pre_proba, pre_label

    
# 使用logistics回归在训练集上训练，并在测试集上进行预测
def logistic_predict(data_train, data_test, best_params):
    clf = LogisticRegression(C=best_params['C'])
    clf.fit(data_train[['price','Distance']], data_train['label'])
    pre_proba = clf.predict_proba(data_test[['price','Distance']])[:,1]
    return pre_proba
    
    
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
    

def save_result(data_test, pre_proba):
    result = data_test[['User_id','Coupon_id','Date_received']].copy()
    result['Probability'] = pre_proba
    result.to_csv('sample_submission.csv', index=False, header=False)
    
    
if __name__ == '__main__':
    data_train, data_test = read_Data()     # 读取数据
    best_params = logistic_tuning(data_train)
#    pre_proba, pre_label = logistic_trainData_predict(data_train, best_params)
    pre_proba = logistic_predict(data_train, data_test, best_params)     # 使用最佳参数进行预测
    save_result(data_test, pre_proba)
#    pre_label = xgboost_predict(data_train)