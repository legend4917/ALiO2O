#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:11:16 2016

@author: zxl
"""


import pandas as pd
from numpy import nan as NA
import string
import datetime
import matplotlib.pyplot as plt
import xgboost as xgb


# 加载数据，并进行预处理
def loadData():
    columns = ['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance', 'Date_received', 'Date']
    data_train = pd.read_csv('ccf_data/ccf_offline_stage1_train.csv', header=None, names=columns, index_col=['Coupon_id'])
    data_test = pd.read_csv('ccf_data/ccf_offline_stage1_test.csv', header=None, names=columns, index_col=['Coupon_id'])
    data_train = data_train.replace({'null':NA})
    return data_train, data_test


# 缺失值处理
def handleNA(data_train):
    data_train = data_train.dropna(subset=['Discount_rate', 'Distance'])    # 直接删除缺失值
    
    
# 将形如 x:y 格式的折扣形式全部转化为小数形式，并讲其价格提取出来作为特征使用
def get_Discount_rate(discount_rate):
    discount_rate_temp = []
    price = []
    for i in range(discount_rate.values.size):
        temp = discount_rate.values[i].split(':')
        if(len(temp) == 2):
            discount_rate_temp.append(round((string.atof(temp[0]) - string.atof(temp[1])) / string.atof(temp[0]), 2))
            price.append(string.atof(temp[0]))
        else:
            discount_rate_temp.append(string.atof(temp[0]))
            price.append(-20)
    return discount_rate_temp, price


# 对距离特征进行归一化处理
def distance_normal(distance):
    return [float(x)/10 for x in distance]
            

def price_normal(price):
    price_min = price.min()
    wid = price.max() - price.min()
    return [(x-price_min) / wid for x in price]
    

# 根据Date_received和Date两个特征得出其所属类别，即优惠券是否在15天内使用
def get_label(Date_received, Date):
    label = []
    flag1 = Date_received.notnull().values
    flag2 = Date.notnull().values
    for i in range(Date.size):
        if(flag1[i] and flag2[i] and string.atoi(Date.values[i],10)-string.atoi(Date_received.values[i],10)<=15):
            date1 = datetime.datetime.strptime(Date.values[i], '%Y%m%d').date()
            date2 = datetime.datetime.strptime(Date_received.values[i], '%Y%m%d').date()
            if((date1 - date2).days <= 15):
                label.append(1)
                continue
        label.append(0)
    return label
    
# 对样本类别进行均衡处理
def get_sepSample(data_train):
    sample_pos = data_train[data_train['label'] == 1]
    sample_neg = data_train[data_train['label'] == 0]
    rate = sample_neg.shape[0] / sample_pos.shape[0]
    temp = sample_neg
    for i in range(rate):
        temp = pd.concat([temp, sample_pos])
    return temp
    
    
    
# 画图查看样本各属性对其类别的影响情况
def tryfind(data_train):
    # 查看Distance对消费券使用情况的影响
    temp = data_train['label'].groupby(data_train['Distance'])
    rate_distance = temp.sum() / temp.size()
    plt.plot(rate_distance, marker='o')
    plt.show()
    
    # 查看Discount_rate对消费券使用情况的影响
    temp = data_train['label'].groupby(data_train['Discount_rate'])
    cnt_discountRate = temp.sum() / temp.size()
    plt.plot(cnt_discountRate.sort_index(), marker='o')
    plt.show()
    
    # 查看商品价格对消费券使用情况的影响
    temp = data_train['label'].groupby(data_train['price'])
    cnt_price = temp.sum() / temp.size()
    cnt_price = cnt_price.drop(-20)
    plt.plot(cnt_price.sort_index(), marker='o')
    
    
