# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: select_user_info_feature.py
    @time: 2020/2/18 14:39
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
import matplotlib.pyplot as plt

# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
# 设置pandas显示文件全部列
pd.set_option("display.max_columns", None)

train_data = pd.read_csv("../data/feature_select/select_train_feature.csv")
user_info_data = pd.read_csv("../data/init/user_info.csv")

# 添加用户基础特征
train_data = pd.merge(train_data, user_info_data, on="user_id", how="left")
# 删除标的成交日期之后插入的用户基础信息
train_data = train_data[train_data.auditing_date > train_data.insertdate]
# 保留标的成交日期之前的最近插入的用户基础信息
union_feature = train_data.groupby(["user_id", "listing_id"])["insertdate"].max().reset_index()
# 根据主属性合并其余附属属性
# user_id + listing_id = train.csv
# user_id + insertdate = user_info.csv
train_data = pd.merge(union_feature, train_data, on=["user_id", "listing_id", "insertdate"], how="left")
train_data["cell_province"] = train_data.apply(
    lambda x: x["id_province"] if x["cell_province"] == r"\N" else x["cell_province"], axis=1
)

# 绘制手机号归属省份-逾期率柱状图
cell_province_default_series = train_data[train_data.label == 1]["cell_province"].value_counts()
cell_province_all_series = train_data["cell_province"].value_counts()
cell_province_default_list = list(cell_province_default_series.keys())
cell_province_default_rate = []
for i in cell_province_default_list:
    cell_province_default_rate.append(cell_province_default_series[i] / cell_province_all_series[i])
fig1 = plt.figure()
tt = plt.bar(cell_province_default_list, cell_province_default_rate)
plt.xlabel(u"手机号归属省份")
plt.ylabel(u"逾期率")
plt.title(u"手机号归属省份-逾期率柱状图")
"""
for b in tt:
    h = b.get_height()
    plt.text(b.get_x()+b.get_width()/2, h, "%.2f" % float(h), ha="center", va="bottom")
"""
plt.xticks(rotation=90)
plt.show()

# 绘制身份证号归属省份-逾期次数柱状图
id_province_default_series = train_data[train_data.label == 1]["id_province"].value_counts()
id_province_all_series = train_data["id_province"].value_counts()
id_province_default_list = list(id_province_default_series.keys())
id_province_default_rate = []
for i in id_province_default_list:
    id_province_default_rate.append(id_province_default_series[i] / id_province_all_series[i])
fig2 = plt.figure()
tt = plt.bar(id_province_default_list, id_province_default_rate)
plt.xlabel(u"身份证号归属省份")
plt.ylabel(u"逾期率")
plt.title(u"身份证号归属省份-逾期率柱状图")
plt.xticks(rotation=90)
plt.show()

# 地域特征对逾期不是很明显，删除地域特征
train_data.drop(["cell_province", "id_province", "id_city"], axis=1, inplace=True)
print("合并后数据行数目:", train_data.shape[0])
print("合并后数据列数目:", train_data.shape[1])
train_data.to_csv("../data/feature_select/select_user_info_feature.csv", index=False)

