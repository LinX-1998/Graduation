# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: select_user_repay_logs_feature2.py
    @time: 2020/4/20 17:45
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
from functools import reduce
from util.calculate_date import calculate_two_day_difference, calculate_before_day
pd.set_option("display.max_columns", None)

repay_data_temp1 = pd.read_csv("../data/feature_select/select_user_repay_logs_feature1_temp1.csv")

# 确定标的成交日期前15，31，90天日期
repay_data_temp1["day_before_15"] = repay_data_temp1["auditing_date"].map(
    lambda x: calculate_before_day(x, 15)
)
repay_data_temp1["day_before_31"] = repay_data_temp1["auditing_date"].map(
    lambda x: calculate_before_day(x, 31)
)
repay_data_temp1["day_before_90"] = repay_data_temp1["auditing_date"].map(
    lambda x: calculate_before_day(x, 90)
)

# 确定标的成交前15，31，90天到期的标的次数
user_deadline_cnt_15 = repay_data_temp1[repay_data_temp1.due_date > repay_data_temp1.day_before_15]\
    .groupby(["user_id", "auditing_date"]).size().reset_index().rename(columns={0: "user_deadline_cnt_15"})
user_deadline_cnt_31 = repay_data_temp1[repay_data_temp1.due_date > repay_data_temp1.day_before_31]\
    .groupby(["user_id", "auditing_date"]).size().reset_index().rename(columns={0: "user_deadline_cnt_31"})
user_deadline_cnt_90 = repay_data_temp1[repay_data_temp1.due_date > repay_data_temp1.day_before_90]\
    .groupby(["user_id", "auditing_date"]).size().reset_index().rename(columns={0: "user_deadline_cnt_90"})

# 确定标的成交前15，31，90天到期的标的总金额
user_deadline_amt_15 = repay_data_temp1[repay_data_temp1.due_date > repay_data_temp1.day_before_15]\
    .groupby(["user_id", "auditing_date"])["due_amt"].sum().reset_index()\
    .rename(columns={"due_amt": "user_deadline_amt_15"})
user_deadline_amt_31 = repay_data_temp1[repay_data_temp1.due_date > repay_data_temp1.day_before_31]\
    .groupby(["user_id", "auditing_date"])["due_amt"].sum().reset_index()\
    .rename(columns={"due_amt": "user_deadline_amt_31"})
user_deadline_amt_90 = repay_data_temp1[repay_data_temp1.due_date > repay_data_temp1.day_before_90]\
    .groupby(["user_id", "auditing_date"])["due_amt"].sum().reset_index()\
    .rename(columns={"due_amt": "user_deadline_amt_90"})

# 确定标的成交前15，31，90天到期的标的逾期次数
user_default_cnt_15 = repay_data_temp1[(repay_data_temp1.repay_date == "2200-01-01") &
                                       (repay_data_temp1.due_date > repay_data_temp1.day_before_15)]\
    .groupby(["user_id", "auditing_date"]).size().reset_index().rename(columns={0: "user_default_cnt_15"})
user_default_cnt_31 = repay_data_temp1[(repay_data_temp1.repay_date == "2200-01-01") &
                                       (repay_data_temp1.due_date > repay_data_temp1.day_before_31)]\
    .groupby(["user_id", "auditing_date"]).size().reset_index().rename(columns={0: "user_default_cnt_31"})
user_default_cnt_90 = repay_data_temp1[(repay_data_temp1.repay_date == "2200-01-01") &
                                       (repay_data_temp1.due_date > repay_data_temp1.day_before_90)]\
    .groupby(["user_id", "auditing_date"]).size().reset_index().rename(columns={0: "user_default_cnt_90"})

# 确定标的成交前15，31，90天到期的标的逾期总金额
user_default_amt_15 = repay_data_temp1[(repay_data_temp1.repay_date == "2200-01-01") &
                                       (repay_data_temp1.due_date > repay_data_temp1.day_before_15)]\
    .groupby(["user_id", "auditing_date"])["due_amt"].sum().reset_index()\
    .rename(columns={"due_amt": "user_default_amt_15"})
user_default_amt_31 = repay_data_temp1[(repay_data_temp1.repay_date == "2200-01-01") &
                                       (repay_data_temp1.due_date > repay_data_temp1.day_before_31)]\
    .groupby(["user_id", "auditing_date"])["due_amt"].sum().reset_index()\
    .rename(columns={"due_amt": "user_default_amt_31"})
user_default_amt_90 = repay_data_temp1[(repay_data_temp1.repay_date == "2200-01-01") &
                                       (repay_data_temp1.due_date > repay_data_temp1.day_before_90)]\
    .groupby(["user_id", "auditing_date"])["due_amt"].sum().reset_index()\
    .rename(columns={"due_amt": "user_default_amt_90"})

# 确定标的成交前最近的还款日以及最远的还款日，以及和标的成交日之间的差距
last_due_date = repay_data_temp1.groupby(["user_id", "auditing_date"])["due_date"].max().reset_index()\
    .rename(columns={"due_date": "last_due_date"})
first_due_date = repay_data_temp1.groupby(["user_id", "auditing_date"])["due_date"].min().reset_index()\
    .rename(columns={"due_date": "first_due_date"})

last_due_date["last_due_date_distance"] = last_due_date.apply(
    lambda x: calculate_two_day_difference(x["last_due_date"], x["auditing_date"]), axis=1)
first_due_date["first_due_date_distance"] = first_due_date.apply(
    lambda x: calculate_two_day_difference(x["first_due_date"], x["auditing_date"]), axis=1)

select_user_repay_logs_feature2 = reduce(lambda x, y: pd.merge(x, y, on=["user_id", "auditing_date"], how="outer"),
                                         [user_deadline_cnt_15, user_deadline_cnt_31, user_deadline_cnt_90,
                                          user_deadline_amt_15, user_deadline_amt_31, user_deadline_amt_90,
                                          user_default_cnt_15, user_default_cnt_31, user_default_cnt_90,
                                          user_default_amt_15, user_default_amt_31, user_default_amt_90,
                                          last_due_date, first_due_date])

select_user_repay_logs_feature2.to_csv("../data/feature_select/select_user_repay_logs_feature2.csv", index=False)
print("合并后数据行数目:", select_user_repay_logs_feature2.shape[0])
print("合并后数据列数目:", select_user_repay_logs_feature2.shape[1])
