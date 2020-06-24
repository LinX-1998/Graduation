# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: dealing_v3.py
    @time: 2020/4/20 19:52
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
import matplotlib.pyplot as plt

# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
# 设置pandas显示文件全部列
pd.set_option("display.max_columns", None)

train_data = pd.read_csv("../data/feature_dealing/dealing_v2.csv")
print("行数目:", train_data.shape[0])
print("列数目:", train_data.shape[1])
ppd_train = train_data.loc[200000:800000-1, :]
ppd_test = train_data.loc[800000:1000000-1, :]
ppd_test = ppd_test.reset_index(drop=True)
ppd_train.to_csv("../data/feature_dealing/dealing_v3_ppd_train.csv", index=False)
ppd_test.to_csv("../data/feature_dealing/dealing_v3_ppd_test.csv", index=False)

# 绘制标的还款结果类别-数据项数目柱状图
label_series = ppd_test["label"].value_counts()
label_list = list(label_series.keys())
label_value = []
for i in label_list:
    label_value.append(label_series[i])
fig1 = plt.figure()
tt = plt.bar([u"未逾期"], [label_value[0]], label=u"未逾期")
tt2 = plt.bar([u"逾期"], [label_value[1]], label=u"逾期")
plt.xlabel(u"标的还款结果")
plt.ylabel(u"数目")
plt.title(u"标的还款结果-数目柱状图")
for b in tt+tt2:
    h = b.get_height()
    plt.text(b.get_x()+b.get_width()/2, h, "%d" % int(h), ha="center", va="bottom")
plt.show()
