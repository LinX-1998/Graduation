# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: select_user_behavior_logs_feature1.py
    @time: 2020/2/19 16:27
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
import matplotlib.pyplot as plt

# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
# 设置pandas显示文件全部列
pd.set_option("display.max_columns", None)

train_data = pd.read_csv("../data/feature_select/select_listing_info_feature.csv", parse_dates=["auditing_date", "due_date"])
behavior_data = pd.read_csv("../data/init/user_behavior_logs.csv", parse_dates=["behavior_time"])
behavior_data = behavior_data.drop_duplicates()
behavior_data["hour"] = (behavior_data["behavior_time"]).dt.hour
behavior_data["day"] = (behavior_data["behavior_time"]).dt.day
behavior_data["month"] = (behavior_data["behavior_time"]).dt.month
behavior_data["year"] = (behavior_data["behavior_time"]).dt.year
behavior_data = pd.merge(train_data[["user_id", "listing_id", "auditing_date", "due_date"]], behavior_data,
                         on=["user_id"], how="left")
behavior_data = behavior_data[behavior_data.due_date > behavior_data.behavior_time]
behavior_data.to_csv("../data/feature_select/select_user_behavior_logs_feature1.csv", index=False)
