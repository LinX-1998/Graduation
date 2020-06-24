# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: dealing_v7.py
    @time: 2020/5/14 17:33
    
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

credit_data = pd.read_csv("../data/init/credit_test.csv")
print(len(credit_data))
credit_data.to_csv("../data/demo/credit_all.csv", index=False)

credit_data_1 = credit_data[0:28481]
print(len(credit_data_1))
credit_data_1.to_csv("../data/demo/credit_default_1.csv", index=False)

credit_data_2 = credit_data[28481:56962]
print(len(credit_data_2))
credit_data_2.to_csv("../data/demo/credit_default_2.csv", index=False)

credit_data_3 = credit_data[56962:85443]
print(len(credit_data_3))
credit_data_3.to_csv("../data/demo/credit_default_3.csv", index=False)

ppd_data = pd.read_csv("../data/feature_dealing/dealing_v4_test.csv")
print(len(ppd_data))
ppd_data.to_csv("../data/demo/ppd_all.csv", index=False)

ppd_data_1 = ppd_data[0:20000]
print(len(ppd_data_1))
ppd_data_1.to_csv("../data/demo/ppd_default_1.csv", index=False)

ppd_data_2 = ppd_data[20000:40000]
print(len(ppd_data_2))
ppd_data_2.to_csv("../data/demo/ppd_default_2.csv", index=False)

ppd_data_3 = ppd_data[40000:60000]
print(len(ppd_data_3))
ppd_data_3.to_csv("../data/demo/ppd_default_3.csv", index=False)

ppd_data_4 = ppd_data[60000:80000]
print(len(ppd_data_4))
ppd_data_4.to_csv("../data/demo/ppd_default_4.csv", index=False)

ppd_data_5 = ppd_data[80000:100000]
print(len(ppd_data_5))
ppd_data_5.to_csv("../data/demo/ppd_default_5.csv", index=False)

ppd_data_6 = ppd_data[100000:120000]
print(len(ppd_data_6))
ppd_data_6.to_csv("../data/demo/ppd_default_6.csv", index=False)

ppd_data_7 = ppd_data[120000:140000]
print(len(ppd_data_7))
ppd_data_7.to_csv("../data/demo/ppd_default_7.csv", index=False)

ppd_data_8 = ppd_data[140000:160000]
print(len(ppd_data_8))
ppd_data_8.to_csv("../data/demo/ppd_default_8.csv", index=False)

ppd_data_9 = ppd_data[160000:170000]
print(len(ppd_data_9))
ppd_data_9.to_csv("../data/demo/ppd_default_9.csv", index=False)

ppd_data_10 = ppd_data[170000:180000]
print(len(ppd_data_10))
ppd_data_10.to_csv("../data/demo/ppd_default_10.csv", index=False)

ppd_data_11 = ppd_data[180000:190000]
print(len(ppd_data_11))
ppd_data_11.to_csv("../data/demo/ppd_default_11.csv", index=False)

ppd_data_12 = ppd_data[190000:200000]
print(len(ppd_data_12))
ppd_data_12.to_csv("../data/demo/ppd_default_12.csv", index=False)
