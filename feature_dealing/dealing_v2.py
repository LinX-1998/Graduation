# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: dealing_v2.py
    @time: 2020/4/20 19:40
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
pd.set_option("display.max_columns", None)

train_data = pd.read_csv("../data/feature_dealing/dealing_v1.csv")
train_data = train_data.sort_values(by=["auditing_date"])
train_data = train_data.reset_index(drop=True)
order = ["label", "user_id", "listing_id", "auditing_date", "due_date", "due_amt", "term", "rate", "principal",
         "month_rate", "gender", "reg_month", "is_age_10", "is_age_20", "is_age_30", "is_age_40", "is_age_50",
         "is_age_60", "is_age_70", "is_age_80", "is_age_90", "is_age_100",
         "day_of_week", "week_of_month", "day_of_month", "month_of_year",
         "do_sum", "user_do_0_3", "user_do_3_6", "user_do_6_9", "user_do_9_12",
         "user_do_12_15", "user_do_15_18", "user_do_18_21", "user_do_21_24",
         "user_do_0_3_rate", "user_do_3_6_rate", "user_do_6_9_rate", "user_do_9_12_rate",
         "user_do_12_15_rate", "user_do_15_18_rate", "user_do_18_21_rate", "user_do_21_24_rate",
         "user_deadline_cnt_15", "user_deadline_cnt_31", "user_deadline_cnt_90",
         "user_deadline_amt_15", "user_deadline_amt_31", "user_deadline_amt_90",
         "user_default_cnt_15", "user_default_cnt_31", "user_default_cnt_90",
         "user_default_amt_15", "user_default_amt_31", "user_default_amt_90",
         "user_success_cnt_15", "user_success_cnt_31", "user_success_cnt_90",
         "user_success_amt_15", "user_success_amt_31", "user_success_amt_90",
         "user_repay_before_15", "user_repay_before_31", "user_repay_before_90",
         "user_repay_amt_before_15", "user_repay_amt_before_31", "user_repay_amt_before_90",
         "last_due_date_distance", "first_due_date_distance", "last_repay_date_distance", "first_repay_date_distance"]
train_data = train_data[order]
train_data.to_csv("../data/feature_dealing/dealing_v2.csv", index=False)
