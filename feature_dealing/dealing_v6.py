# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: dealing_v6.py
    @time: 2020/5/7 7:14
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
# 设置pandas显示文件全部列
pd.set_option("display.max_columns", None)

all_data = pd.read_csv("../data/init/creditcard.csv")
print("测试集数目:", len(all_data))
print("政府类样本数目: %s" % Counter(all_data["Class"]))

train_data = pd.read_csv("../data/init/credit_train.csv")
print("测试集数目:", len(train_data))
print("政府类样本数目: %s" % Counter(train_data["Class"]))

test_data = pd.read_csv("../data/init/credit_test.csv")
print("测试集数目:", len(test_data))
print("政府类样本数目: %s" % Counter(test_data["Class"]))
