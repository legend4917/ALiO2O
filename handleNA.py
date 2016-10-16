#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 15:12:15 2016

@author: zxl
"""

import numpy as np


def handle_trainDistanceNA(data_train):
    distance_median = data_train['Distance'].median(skipna=True)    # 所有Distance的中位数
    for user_id, group in data_train['Distance'].groupby(data_train['User_id']):
        if (group.isnull().any()):
            group_distance_median = group.median(skipna=True)   # 当前group的中位数
            if (np.isnan(group_distance_median)):
                group_distance_median = distance_median     # 如果当前group的Distance全为缺失值，这设为distance_median
#            print group_distance_median
            index = list(group[group.isnull()].index)
#            print '====================================='
#            print data_train['Distance'][index]
#            print data_train['Distance'][index]._is_view
            data_train['Distance'][index] = group_distance_median
#            print data_train['Distance'][index]

#    dataTrain_temp = data_train.set_index(['User_id'])
#    for id in set(data_train['User_id']):
#        print '====================================='
#        temp = dataTrain_temp.ix[id]
#        print type(temp)
    return data_train