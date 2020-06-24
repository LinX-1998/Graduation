# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: select_user_repay_logs_feature3.py
    @time: 2020/4/20 17:52
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
from functools import reduce
from util.calculate_date import calculate_two_day_difference, calculate_before_day
pd.set_option("display.max_columns", None)

repay_data_temp2 = pd.read_csv("../data/feature_select/select_user_repay_logs_feature1_temp2.csv")

# 确定标的成交日期前15，31，90天日期
repay_data_temp2["day_before_15"] = repay_data_temp2["auditing_date"].map(
    lambda x: calculate_before_day(x, 15)
)
repay_data_temp2["day_before_31"] = repay_data_temp2["auditing_date"].map(
    lambda x: calculate_before_day(x, 31)
)
repay_data_temp2["day_before_90"] = repay_data_temp2["auditing_date"].map(
    lambda x: calculate_before_day(x, 90)
)

# 确定标的成交前15，31，90天到期的标的次数
user_success_cnt_15 = repay_data_temp2[repay_data_temp2.repay_date > repay_data_temp2.day_before_15]\
    .groupby(["user_id", "auditing_date"]).size().reset_index().rename(columns={0: "user_success_cnt_15"})
user_success_cnt_31 = repay_data_temp2[repay_data_temp2.repay_date > repay_data_temp2.day_before_31]\
    .groupby(["user_id", "auditing_date"]).size().reset_index().rename(columns={0: "user_success_cnt_31"})
user_success_cnt_90 = repay_data_temp2[repay_data_temp2.repay_date > repay_data_temp2.day_before_90]\
    .groupby(["user_id", "auditing_date"]).size().reset_index().rename(columns={0: "user_success_cnt_90"})

# 确定标的成交前15，31，90天到期的标的总金额
user_success_amt_15 = repay_data_temp2[repay_data_temp2.repay_date > repay_data_temp2.day_before_15]\
    .groupby(["user_id", "auditing_date"])["repay_amt"].sum().reset_index()\
    .rename(columns={"repay_amt": "user_success_amt_15"})
user_success_amt_31 = repay_data_temp2[repay_data_temp2.repay_date > repay_data_temp2.day_before_31]\
    .groupby(["user_id", "auditing_date"])["repay_amt"].sum().reset_index()\
    .rename(columns={"repay_amt": "user_success_amt_31"})
user_success_amt_90 = repay_data_temp2[repay_data_temp2.repay_date > repay_data_temp2.day_before_90]\
    .groupby(["user_id", "auditing_date"])["repay_amt"].sum().reset_index()\
    .rename(columns={"repay_amt": "user_success_amt_90"})

# 用户提前还款的单数
repay_data_temp2_before = repay_data_temp2[repay_data_temp2.back_days > 0]

user_repay_before_15 = repay_data_temp2_before[
    repay_data_temp2_before.repay_date > repay_data_temp2_before.day_before_15].groupby(
    ["user_id", "auditing_date"]).size()\
    .reset_index().rename(columns={0: "user_repay_before_15"})
user_repay_before_31 = repay_data_temp2_before[
    repay_data_temp2_before.repay_date > repay_data_temp2_before.day_before_31].groupby(
    ["user_id", "auditing_date"]).size() \
    .reset_index().rename(columns={0: "user_repay_before_31"})
user_repay_before_90 = repay_data_temp2_before[
    repay_data_temp2_before.repay_date > repay_data_temp2_before.day_before_90].groupby(
    ["user_id", "auditing_date"]).size() \
    .reset_index().rename(columns={0: "user_repay_before_90"})

# 用户提前还款的金额
user_repay_amt_before_15 = repay_data_temp2_before[
    repay_data_temp2_before.repay_date > repay_data_temp2_before.day_before_15].groupby(
    ["user_id", "auditing_date"])["due_amt"].sum() \
    .reset_index().rename(columns={"due_amt": "user_repay_amt_before_15"})
user_repay_amt_before_31 = repay_data_temp2_before[
    repay_data_temp2_before.repay_date > repay_data_temp2_before.day_before_31].groupby(
    ["user_id", "auditing_date"])["due_amt"].sum() \
    .reset_index().rename(columns={"due_amt": "user_repay_amt_before_31"})
user_repay_amt_before_90 = repay_data_temp2_before[
    repay_data_temp2_before.repay_date > repay_data_temp2_before.day_before_90].groupby(
    ["user_id", "auditing_date"])["due_amt"].sum() \
    .reset_index().rename(columns={"due_amt": "user_repay_amt_before_90"})

# 最后和最早付款日
last_repay_date = repay_data_temp2.groupby(["user_id", "auditing_date"])["repay_date"].max().reset_index().rename(
    columns={"repay_date": "last_repay_date"})
first_repay_date = repay_data_temp2.groupby(["user_id", "auditing_date"])["repay_date"].min().reset_index().rename(
    columns={"repay_date": "first_repay_date"})
last_repay_date["last_repay_date_distance"] = last_repay_date.apply(
    lambda x: calculate_two_day_difference(x["last_repay_date"], x["auditing_date"]), axis=1)
first_repay_date["first_repay_date_distance"] = first_repay_date.apply(
    lambda x: calculate_two_day_difference(x["first_repay_date"], x["auditing_date"]), axis=1)

select_user_repay_logs_feature2 = reduce(lambda x, y: pd.merge(x, y, on=["user_id", "auditing_date"], how="outer"),
                                         [user_success_cnt_15, user_success_cnt_31, user_success_cnt_90,
                                          user_success_amt_15, user_success_amt_31, user_success_amt_90,
                                          user_repay_before_15, user_repay_before_31, user_repay_before_90,
                                          user_repay_amt_before_15, user_repay_amt_before_31, user_repay_amt_before_90,
                                          last_repay_date, first_repay_date])

select_user_repay_logs_feature2.to_csv("../data/feature_select/select_user_repay_logs_feature3.csv", index=False)
print("合并后数据行数目:", select_user_repay_logs_feature2.shape[0])
print("合并后数据列数目:", select_user_repay_logs_feature2.shape[1])
