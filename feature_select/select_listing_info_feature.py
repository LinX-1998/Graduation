# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: select_listing_info_feature.py
    @time: 2020/2/18 16:13
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
import matplotlib.pyplot as plt

# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
# 设置pandas显示文件全部列
pd.set_option("display.max_columns", None)

train_data = pd.read_csv("../data/feature_select/select_user_info_feature.csv", parse_dates=["due_date"])
# 标的成交日期已经确定，删除auditing_date列
listing_info_data = pd.read_csv("../data/init/listing_info.csv", usecols=["user_id", "listing_id",
                                                                          "term", "rate", "principal"])

# user_id + listing_id = listing_info.csv
train_data = pd.merge(train_data, listing_info_data, on=["user_id", "listing_id"], how="left")
train_data["month_rate"] = train_data.apply(lambda x: x["rate"] / x["term"], axis=1)
# 增加还款日期星期几，几月，几号，本月第几周
train_data["day_of_week"] = (train_data["due_date"]).dt.dayofweek
train_data["month_of_year"] = (train_data["due_date"]).dt.month
train_data["day_of_month"] = (train_data["due_date"]).dt.day
train_data["week_of_month"] = (train_data["due_date"]).dt.day//7
print("合并后数据行数目:", train_data.shape[0])
print("合并后数据列数目:", train_data.shape[1])
train_data.to_csv("../data/feature_select/select_listing_info_feature.csv", index=False)
