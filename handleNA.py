#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 15:12:15 2016

@author: zxl
"""

import numpy as np
from scipy import stats


def fillTrainDataNA_withGroupMode(data_train):
    distance_mode = stats.mode(data_train['Distance']).mode[0]    # 所有Distance的中位数
    for user_id, group in data_train['Distance'].groupby(data_train['User_id']):
        if (group.isnull().any()):
            group_distance_mode = stats.mode(group).mode(0)   # 当前group的中位数
            print type(group_distance_mode)
#            if (np.isnan(group_distance_mode)):
#                group_distance_mode = distance_mode     # 如果当前group的Distance全为缺失值，这设为distance_median
            index = list(group[group.isnull()].index)
#            print '====================================='
#            print data_train['Distance'][index]
#            print data_train['Distance'][index]._is_view
            data_train['Distance'][index] = group_distance_mode
#            print data_train['Distance'][index]

#    dataTrain_temp = data_train.set_index(['User_id'])
#    for id in set(data_train['User_id']):
#        print '====================================='
#        temp = dataTrain_temp.ix[id]
#        print type(temp)
    return data_train
    


# 训练数据缺失值处理
def removeAllNA(data_train):
    data_train = data_train.dropna(subset=['Coupon_id','Discount_rate', 'Distance'])    # 直接删除缺失值
    return data_train
    

# 测试数据缺失值处理
def fillTestDataNA(data_train, data_test):
    dataTrain_median = data_train['Distance'].median(skipna=True)
    data_train = data_train.set_index(['User_id'])
    data_testNA = data_test[data_test['Distance'].isnull()]
    for index in data_testNA.index:
        user_id = data_testNA.ix[index]['User_id']
        if user_id in data_train.index:
            group = data_train.ix[user_id]['Distance']
            data_test['Distance'][index] = np.median(group)
            
        else:
            data_test['Distance'][index] = dataTrain_median
    return data_test
    
    
# 测试数据集缺失值处理：使用众数代替
def fillTestDataNA_withMode(data_train, data_test):
    dataTrain_mode = stats.mode(data_train['Distance']).mode[0]
    return data_test.fillna(dataTrain_mode)