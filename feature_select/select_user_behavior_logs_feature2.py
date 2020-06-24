# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: select_user_behavior_logs_feature2.py
    @time: 2020/2/19 15:48
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
# 设置pandas显示文件全部列
pd.set_option("display.max_columns", None)

train_data = pd.read_csv("../data/feature_select/select_listing_info_feature.csv")
behavior_data = pd.read_csv("../data/feature_select/select_user_behavior_logs_feature1.csv",
                            parse_dates=["behavior_time"])

user_do_0_3 = behavior_data[(behavior_data.hour >= 0) & (behavior_data.hour < 3)].drop_duplicates(subset=["user_id", "listing_id", "year", "month", "day"], keep="first")\
    .groupby(["user_id", "listing_id"]).size().reset_index().rename(columns={0: "user_do_0_3"})
# train_data = pd.merge(train_data, user_do_0_3, on=["user_id", "listing_id"], how="left")

user_do_3_6 = behavior_data[(behavior_data.hour >= 3) & (behavior_data.hour < 6)].drop_duplicates(subset=["user_id", "listing_id", "year", "month", "day"], keep="first")\
    .groupby(["user_id", "listing_id"]).size().reset_index().rename(columns={0: "user_do_3_6"})
# train_data = pd.merge(train_data, user_do_3_6, on=["user_id", "listing_id"], how="left")

user_do_6_9 = behavior_data[(behavior_data.hour >= 6) & (behavior_data.hour < 9)].drop_duplicates(subset=["user_id", "listing_id", "year", "month", "day"], keep="first")\
    .groupby(["user_id", "listing_id"]).size().reset_index().rename(columns={0: "user_do_6_9"})
# train_data = pd.merge(train_data, user_do_6_9, on=["user_id", "listing_id"], how="left")

user_do_9_12 = behavior_data[(behavior_data.hour >= 9) & (behavior_data.hour < 12)].drop_duplicates(subset=["user_id", "listing_id", "year", "month", "day"], keep="first")\
    .groupby(["user_id", "listing_id"]).size().reset_index().rename(columns={0: "user_do_9_12"})
# train_data = pd.merge(train_data, user_do_9_12, on=["user_id", "listing_id"], how="left")

user_do_12_15 = behavior_data[(behavior_data.hour >= 12) & (behavior_data.hour < 15)].drop_duplicates(subset=["user_id", "listing_id", "year", "month", "day"], keep="first")\
    .groupby(["user_id", "listing_id"]).size().reset_index().rename(columns={0: "user_do_12_15"})
# train_data = pd.merge(train_data, user_do_12_15, on=["user_id", "listing_id"], how="left")

user_do_15_18 = behavior_data[(behavior_data.hour >= 15) & (behavior_data.hour < 18)].drop_duplicates(subset=["user_id", "listing_id", "year", "month", "day"], keep="first")\
    .groupby(["user_id", "listing_id"]).size().reset_index().rename(columns={0: "user_do_15_18"})
# train_data = pd.merge(train_data, user_do_15_18, on=["user_id", "listing_id"], how="left")

user_do_18_21 = behavior_data[(behavior_data.hour >= 18) & (behavior_data.hour < 21)].drop_duplicates(subset=["user_id", "listing_id", "year", "month", "day"], keep="first")\
    .groupby(["user_id", "listing_id"]).size().reset_index().rename(columns={0: "user_do_18_21"})
# train_data = pd.merge(train_data, user_do_18_21, on=["user_id", "listing_id"], how="left")

user_do_21_24 = behavior_data[(behavior_data.hour >= 21) & (behavior_data.hour < 24)].drop_duplicates(subset=["user_id", "listing_id", "year", "month", "day"], keep="first")\
    .groupby(["user_id", "listing_id"]).size().reset_index().rename(columns={0: "user_do_21_24"})
# train_data = pd.merge(train_data, user_do_21_24, on=["user_id", "listing_id"], how="left")

select_user_repay_logs_feature2 = reduce(lambda x, y: pd.merge(x, y, on=["user_id", "listing_id"], how="outer"),
                                         [user_do_0_3, user_do_3_6, user_do_6_9, user_do_9_12,
                                          user_do_12_15, user_do_15_18, user_do_18_21, user_do_21_24])
train_data = pd.merge(train_data, select_user_repay_logs_feature2, on=["user_id", "listing_id"], how="left")
train_data = train_data.fillna(0)
train_data.to_csv("../data/feature_select/select_user_behavior_logs_feature2.csv", index=False)
