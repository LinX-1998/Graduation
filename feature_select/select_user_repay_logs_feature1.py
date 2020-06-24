# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: select_user_repay_logs_feature1.py
    @time: 2020/4/20 17:37
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
from util.calculate_date import calculate_two_day_difference
pd.set_option("display.max_columns", None)

train_data = pd.read_csv("../data/feature_select/select_user_behavior_logs_feature2.csv")
repay_data = pd.read_csv("../data/init/user_repay_logs.csv")

# 记录提前还款天数
repay_data["back_days"] = repay_data.apply(lambda x: calculate_two_day_difference(x["repay_date"], x["due_date"]), axis=1)

# 统筹和训练集有关的还款日志信息
repay_data_temp = pd.merge(train_data[["user_id", "listing_id", "auditing_date"]], repay_data, on=["user_id"], how="left")

# 用户历史标的到期日在此标的成交日之前的数据
repay_data_temp1 = repay_data_temp[repay_data_temp.auditing_date > repay_data_temp.due_date]

# 用户历史标的还款日在此标的成交日之前的数据
repay_data_temp2 = repay_data_temp[repay_data_temp.auditing_date > repay_data_temp.repay_date]
repay_data_temp1.to_csv("../data/feature_select/select_user_repay_logs_feature1_temp1.csv", index=None)
print("合并后数据行数目:", repay_data_temp1.shape[0])
print("合并后数据列数目:", repay_data_temp1.shape[1])
repay_data_temp2.to_csv("../data/feature_select/select_user_repay_logs_feature1_temp2.csv", index=None)
print("合并后数据行数目:", repay_data_temp2.shape[0])
print("合并后数据列数目:", repay_data_temp2.shape[1])
