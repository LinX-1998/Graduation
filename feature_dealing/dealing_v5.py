# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: dealing_v5.py
    @time: 2020/5/2 13:41
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
# 设置pandas显示文件全部列
pd.set_option("display.max_columns", None)
"""
train_data = pd.read_csv("../data/feature_dealing/dealing_v2.csv")
is_age = []
is_age1 = []
is_age2 = []
is_name = ["10-20", "20-30", "30-40", "40-50", "50-60"]
train_data_temp = train_data[train_data.label == 1]
for i in range(1, 6):
    age_name = "is_age_" + str(i*10+10)
    num_dict1 = train_data[age_name].value_counts()
    num_dict2 = train_data_temp[age_name].value_counts()
    try:
        is_age1.append(num_dict1[1])
    except:
        is_age1.append(0)
    try:
        is_age2.append(num_dict2[1])
    except:
        is_age2.append(0)
    if is_age1[i-1] == 0:
        is_age.append(0)
    else:
        is_age.append(is_age2[i-1]/is_age1[i-1])

fig1 = plt.figure()
tt = plt.bar(is_name, is_age2)
plt.xlabel(u"年龄")
plt.ylabel(u"逾期人数")
plt.title(u"年龄段-逾期人数柱状图")
for b in tt:
    h = b.get_height()
    plt.text(b.get_x()+b.get_width()/2, h, "%d" % int(h), ha="center", va="bottom")
plt.show()
"""


def sigmoid(z):
    return 1/(1 + np.exp(-z))


z = np.arange(-9.5, 10, 0.1)
phi_z = sigmoid(z)

plt.figure(figsize=(9, 6))
plt.plot(z, phi_z)
plt.axvline(0, c='black')
plt.axhspan(.0, 1.0, facecolor='0.93', alpha=1.0, ls=':', edgecolor='0.4')
plt.axhline(y=.5, color='.3',alpha=1.0, ls=':')
plt.yticks([.0, .5, 1.0])
plt.yticks([.0, .5, 1.0])
plt.ylim(-.1, 1.1)
plt.title('sigmoid 函数图形化', fontsize=23)
plt.xlabel('z', fontsize=19)
plt.ylabel('f(z)', fontsize=13)
plt.show()
print()
