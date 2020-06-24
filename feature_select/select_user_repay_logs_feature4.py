# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: select_user_repay_logs_feature4.py
    @time: 2020/4/20 18:01
    
    @introduce: Just a __init__.py file
"""
import pandas as pd

select_user_repay_logs_feature2 = pd.read_csv("../data/feature_select/select_user_repay_logs_feature2.csv")
select_user_repay_logs_feature3 = pd.read_csv("../data/feature_select/select_user_repay_logs_feature3.csv")
train_data = pd.read_csv("../data/feature_select/select_user_behavior_logs_feature2.csv")

train_data = pd.merge(train_data, select_user_repay_logs_feature2, on=["user_id", "auditing_date"], how="left")
train_data = pd.merge(train_data, select_user_repay_logs_feature3, on=["user_id", "auditing_date"], how="left")
print("合并后数据行数目:", train_data.shape[0])
print("合并后数据列数目:", train_data.shape[1])
train_data["gender"] = train_data["gender"].apply(lambda x: 1 if x == u"男" else 0)
train_data["do_sum"] = train_data.apply(lambda x: x["user_do_0_3"] + x["user_do_3_6"] + x["user_do_6_9"] +
                                        x["user_do_9_12"] + x["user_do_12_15"] + x["user_do_15_18"] +
                                        x["user_do_18_21"] + x["user_do_21_24"], axis=1)
train_data = train_data.fillna(0)
train_data.to_csv("../data/feature_select/select_user_repay_logs_feature4.csv", index=False)
