# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: dealing_v1.py
    @time: 2020/4/20 19:26
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
from util.calculate_date import calculate_two_month_difference, calculate_two_day_difference

pd.set_option("display.max_columns", None)

train_data = pd.read_csv("../data/feature_select/select_user_repay_logs_feature4.csv")
train_data["reg_month"] = train_data.apply(lambda x: calculate_two_month_difference(x["reg_mon"], x["auditing_date"]), axis=1)

train_data["is_age_10"] = train_data.apply(lambda x: 1 if x["age"] // 10 * 10 + 10 == 10 else 0, axis=1)
train_data["is_age_20"] = train_data.apply(lambda x: 1 if x["age"] // 10 * 10 + 10 == 20 else 0, axis=1)
train_data["is_age_30"] = train_data.apply(lambda x: 1 if x["age"] // 10 * 10 + 10 == 30 else 0, axis=1)
train_data["is_age_40"] = train_data.apply(lambda x: 1 if x["age"] // 10 * 10 + 10 == 40 else 0, axis=1)
train_data["is_age_50"] = train_data.apply(lambda x: 1 if x["age"] // 10 * 10 + 10 == 50 else 0, axis=1)
train_data["is_age_60"] = train_data.apply(lambda x: 1 if x["age"] // 10 * 10 + 10 == 60 else 0, axis=1)
train_data["is_age_70"] = train_data.apply(lambda x: 1 if x["age"] // 10 * 10 + 10 == 70 else 0, axis=1)
train_data["is_age_80"] = train_data.apply(lambda x: 1 if x["age"] // 10 * 10 + 10 == 80 else 0, axis=1)
train_data["is_age_90"] = train_data.apply(lambda x: 1 if x["age"] // 10 * 10 + 10 == 90 else 0, axis=1)
train_data["is_age_100"] = train_data.apply(lambda x: 1 if x["age"] // 10 * 10 + 10 == 100 else 0, axis=1)

train_data["user_do_0_3_rate"] = train_data.apply(lambda x: 0 if x["do_sum"] == 0 else x["user_do_0_3"] / x["do_sum"], axis=1)
train_data["user_do_3_6_rate"] = train_data.apply(lambda x: 0 if x["do_sum"] == 0 else x["user_do_3_6"] / x["do_sum"], axis=1)
train_data["user_do_6_9_rate"] = train_data.apply(lambda x: 0 if x["do_sum"] == 0 else x["user_do_6_9"] / x["do_sum"], axis=1)
train_data["user_do_9_12_rate"] = train_data.apply(lambda x: 0 if x["do_sum"] == 0 else x["user_do_9_12"] / x["do_sum"], axis=1)
train_data["user_do_12_15_rate"] = train_data.apply(lambda x: 0 if x["do_sum"] == 0 else x["user_do_12_15"] / x["do_sum"], axis=1)
train_data["user_do_15_18_rate"] = train_data.apply(lambda x: 0 if x["do_sum"] == 0 else x["user_do_15_18"] / x["do_sum"], axis=1)
train_data["user_do_18_21_rate"] = train_data.apply(lambda x: 0 if x["do_sum"] == 0 else x["user_do_18_21"] / x["do_sum"], axis=1)
train_data["user_do_21_24_rate"] = train_data.apply(lambda x: 0 if x["do_sum"] == 0 else x["user_do_21_24"] / x["do_sum"], axis=1)
train_data = train_data.drop(
    ["insertdate", "age", "repay_date", "repay_amt", "reg_mon",
     "last_due_date", "first_due_date", "last_repay_date", "first_repay_date"], axis=1)

train_data.to_csv("../data/feature_dealing/dealing_v1.csv", index=False)
print(train_data.shape[0])
print(train_data.shape[1])
