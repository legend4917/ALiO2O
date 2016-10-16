#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 09:46:56 2016

@author: zxl
"""

import pandas as pd
from DataPretreatment import *


def save_data(data_trainNotNA, data_testNotNA):
    data_trainNotNA.to_csv('data_trainNotNA.csv', na_rep='null')
    data_testNotNA.to_csv('data_testNotNA.csv')


def read_Data():
    data_train = pd.read_csv('data_trainNotNA.csv')
#    data_train['Receive_date'].replace({'null', NA})
    data_test = pd.read_csv('data_testNotNA.csv')
    return data_train, data_test
    
    
if __name__ == '__main__':
    data_train, data_test = dataPretreatment()
    save_data(data_train, data_test)