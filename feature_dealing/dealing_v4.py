# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: dealing_v4.py
    @time: 2020/4/20 19:56
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)

train_data = pd.read_csv("../data/feature_dealing/dealing_v3_ppd_train.csv")
print("行数目:", train_data.shape[0])
print("列数目:", train_data.shape[1])
test_data = pd.read_csv("../data/feature_dealing/dealing_v3_ppd_test.csv")
print("行数目:", test_data.shape[0])
print("列数目:", test_data.shape[1])

t_max1 = np.max(train_data["due_amt"])
t_mean1 = np.mean(train_data["due_amt"])
t_min1 = np.min(train_data["due_amt"])
train_data["due_amt"] = train_data["due_amt"].apply(lambda x: (x-t_mean1)/(t_max1-t_min1))
test_data["due_amt"] = test_data["due_amt"].apply(lambda x: (x-t_mean1)/(t_max1-t_min1))

t_max2 = np.max(train_data["term"])
t_mean2 = np.mean(train_data["term"])
t_min2 = np.min(train_data["term"])
train_data["term"] = train_data["term"].apply(lambda x: (x-t_mean2)/(t_max2-t_min2))
test_data["term"] = test_data["term"].apply(lambda x: (x-t_mean2)/(t_max2-t_min2))

t_max3 = np.max(train_data["rate"])
t_mean3 = np.mean(train_data["rate"])
t_min3 = np.min(train_data["rate"])
train_data["rate"] = train_data["rate"].apply(lambda x: (x-t_mean3)/(t_max3-t_min3))
test_data["rate"] = test_data["rate"].apply(lambda x: (x-t_mean3)/(t_max3-t_min3))

t_max4 = np.max(train_data["principal"])
t_mean4 = np.mean(train_data["principal"])
t_min4 = np.min(train_data["principal"])
train_data["principal"] = train_data["principal"].apply(lambda x: (x-t_mean4)/(t_max4-t_min4))
test_data["principal"] = test_data["principal"].apply(lambda x: (x-t_mean4)/(t_max4-t_min4))

t_max5 = np.max(train_data["month_rate"])
t_mean5 = np.mean(train_data["month_rate"])
t_min5 = np.min(train_data["month_rate"])
train_data["month_rate"] = train_data["month_rate"].apply(lambda x: (x-t_mean5)/(t_max5-t_min5))
test_data["month_rate"] = test_data["month_rate"].apply(lambda x: (x-t_mean5)/(t_max5-t_min5))

t_max6 = np.max(train_data["reg_month"])
t_mean6 = np.mean(train_data["reg_month"])
t_min6 = np.min(train_data["reg_month"])
train_data["month_rate"] = train_data["reg_month"].apply(lambda x: (x-t_mean6)/(t_max6-t_min6))
test_data["reg_month"] = test_data["reg_month"].apply(lambda x: (x-t_mean6)/(t_max6-t_min6))

t_max7 = np.max(train_data["day_of_week"])
t_mean7 = np.mean(train_data["day_of_week"])
t_min7 = np.min(train_data["day_of_week"])
train_data["day_of_week"] = train_data["day_of_week"].apply(lambda x: (x-t_mean7)/(t_max7-t_min7))
test_data["day_of_week"] = test_data["day_of_week"].apply(lambda x: (x-t_mean7)/(t_max7-t_min7))

t_max8 = np.max(train_data["week_of_month"])
t_mean8 = np.mean(train_data["week_of_month"])
t_min8 = np.min(train_data["week_of_month"])
train_data["week_of_month"] = train_data["week_of_month"].apply(lambda x: (x-t_mean8)/(t_max8-t_min8))
test_data["week_of_month"] = test_data["week_of_month"].apply(lambda x: (x-t_mean8)/(t_max8-t_min8))

t_max9 = np.max(train_data["day_of_month"])
t_mean9 = np.mean(train_data["day_of_month"])
t_min9 = np.min(train_data["day_of_month"])
train_data["day_of_month"] = train_data["day_of_month"].apply(lambda x: (x-t_mean9)/(t_max9-t_min9))
test_data["day_of_month"] = test_data["day_of_month"].apply(lambda x: (x-t_mean9)/(t_max9-t_min9))

t_max10 = np.max(train_data["month_of_year"])
t_mean10 = np.mean(train_data["month_of_year"])
t_min10 = np.min(train_data["month_of_year"])
train_data["month_rate"] = train_data["month_of_year"].apply(lambda x: (x-t_mean10)/(t_max10-t_min10))
test_data["month_of_year"] = test_data["month_of_year"].apply(lambda x: (x-t_mean10)/(t_max10-t_min10))

t_max11 = np.max(train_data["user_deadline_cnt_15"])
t_mean11 = np.mean(train_data["user_deadline_cnt_15"])
t_min11 = np.min(train_data["user_deadline_cnt_15"])
train_data["user_deadline_cnt_15"] = train_data["user_deadline_cnt_15"].apply(lambda x: (x-t_mean11)/(t_max11-t_min11))
test_data["user_deadline_cnt_15"] = test_data["user_deadline_cnt_15"].apply(lambda x: (x-t_mean11)/(t_max11-t_min11))

t_max12 = np.max(train_data["user_deadline_cnt_31"])
t_mean12 = np.mean(train_data["user_deadline_cnt_31"])
t_min12 = np.min(train_data["user_deadline_cnt_31"])
train_data["user_deadline_cnt_31"] = train_data["user_deadline_cnt_31"].apply(lambda x: (x-t_mean12)/(t_max12-t_min12))
test_data["user_deadline_cnt_31"] = test_data["user_deadline_cnt_31"].apply(lambda x: (x-t_mean12)/(t_max12-t_min12))

t_max13 = np.max(train_data["user_deadline_cnt_90"])
t_mean13 = np.mean(train_data["user_deadline_cnt_90"])
t_min13 = np.min(train_data["user_deadline_cnt_90"])
train_data["user_deadline_cnt_90"] = train_data["user_deadline_cnt_90"].apply(lambda x: (x-t_mean13)/(t_max13-t_min13))
test_data["user_deadline_cnt_90"] = test_data["user_deadline_cnt_90"].apply(lambda x: (x-t_mean13)/(t_max13-t_min13))

t_max14 = np.max(train_data["user_deadline_amt_15"])
t_mean14 = np.mean(train_data["user_deadline_amt_15"])
t_min14 = np.min(train_data["user_deadline_amt_15"])
train_data["user_deadline_amt_15"] = train_data["user_deadline_amt_15"].apply(lambda x: (x-t_mean14)/(t_max14-t_min14))
test_data["user_deadline_amt_15"] = test_data["user_deadline_amt_15"].apply(lambda x: (x-t_mean14)/(t_max14-t_min14))

t_max15 = np.max(train_data["user_deadline_amt_31"])
t_mean15 = np.mean(train_data["user_deadline_amt_31"])
t_min15 = np.min(train_data["user_deadline_amt_31"])
train_data["user_deadline_amt_31"] = train_data["user_deadline_amt_31"].apply(lambda x: (x-t_mean15)/(t_max15-t_min15))
test_data["user_deadline_amt_31"] = test_data["user_deadline_amt_31"].apply(lambda x: (x-t_mean15)/(t_max15-t_min15))

t_max16 = np.max(train_data["user_deadline_amt_90"])
t_mean16 = np.mean(train_data["user_deadline_amt_90"])
t_min16 = np.min(train_data["user_deadline_amt_90"])
train_data["user_deadline_amt_90"] = train_data["user_deadline_amt_90"].apply(lambda x: (x-t_mean16)/(t_max16-t_min16))
test_data["user_deadline_amt_90"] = test_data["user_deadline_amt_90"].apply(lambda x: (x-t_mean16)/(t_max16-t_min16))

t_max17 = np.max(train_data["user_default_cnt_15"])
t_mean17 = np.mean(train_data["user_default_cnt_15"])
t_min17 = np.min(train_data["user_default_cnt_15"])
train_data["user_default_cnt_15"] = train_data["user_default_cnt_15"].apply(lambda x: (x-t_mean17)/(t_max17-t_min17))
test_data["user_default_cnt_15"] = test_data["user_default_cnt_15"].apply(lambda x: (x-t_mean17)/(t_max17-t_min17))

t_max18 = np.max(train_data["user_default_cnt_31"])
t_mean18 = np.mean(train_data["user_default_cnt_31"])
t_min18 = np.min(train_data["user_default_cnt_31"])
train_data["user_default_cnt_31"] = train_data["user_default_cnt_31"].apply(lambda x: (x-t_mean18)/(t_max18-t_min18))
test_data["user_default_cnt_31"] = test_data["user_default_cnt_31"].apply(lambda x: (x-t_mean18)/(t_max18-t_min18))

t_max19 = np.max(train_data["user_default_cnt_90"])
t_mean19 = np.mean(train_data["user_default_cnt_90"])
t_min19 = np.min(train_data["user_default_cnt_90"])
train_data["user_default_cnt_90"] = train_data["user_default_cnt_90"].apply(lambda x: (x-t_mean19)/(t_max19-t_min19))
test_data["user_default_cnt_90"] = test_data["user_default_cnt_90"].apply(lambda x: (x-t_mean19)/(t_max19-t_min19))

t_max20 = np.max(train_data["user_default_amt_15"])
t_mean20 = np.mean(train_data["user_default_amt_15"])
t_min20 = np.min(train_data["user_default_amt_15"])
train_data["user_default_amt_15"] = train_data["user_default_amt_15"].apply(lambda x: (x-t_mean20)/(t_max20-t_min20))
test_data["user_default_amt_15"] = test_data["user_default_amt_15"].apply(lambda x: (x-t_mean20)/(t_max20-t_min20))

t_max21 = np.max(train_data["user_default_amt_31"])
t_mean21 = np.mean(train_data["user_default_amt_31"])
t_min21 = np.min(train_data["user_default_amt_31"])
train_data["user_default_amt_31"] = train_data["user_default_amt_31"].apply(lambda x: (x-t_mean21)/(t_max21-t_min21))
test_data["user_default_amt_31"] = test_data["user_default_amt_31"].apply(lambda x: (x-t_mean21)/(t_max21-t_min21))

t_max22 = np.max(train_data["user_default_amt_90"])
t_mean22 = np.mean(train_data["user_default_amt_90"])
t_min22 = np.min(train_data["user_default_amt_90"])
train_data["user_default_amt_90"] = train_data["user_default_amt_90"].apply(lambda x: (x-t_mean22)/(t_max22-t_min22))
test_data["user_default_amt_90"] = test_data["user_default_amt_90"].apply(lambda x: (x-t_mean22)/(t_max22-t_min22))

t_max23 = np.max(train_data["user_success_cnt_15"])
t_mean23 = np.mean(train_data["user_success_cnt_15"])
t_min23 = np.min(train_data["user_success_cnt_15"])
train_data["user_success_cnt_15"] = train_data["user_success_cnt_15"].apply(lambda x: (x-t_mean23)/(t_max23-t_min23))
test_data["user_success_cnt_15"] = test_data["user_success_cnt_15"].apply(lambda x: (x-t_mean23)/(t_max23-t_min23))

t_max24 = np.max(train_data["user_success_cnt_31"])
t_mean24 = np.mean(train_data["user_success_cnt_31"])
t_min24 = np.min(train_data["user_success_cnt_31"])
train_data["user_success_cnt_31"] = train_data["user_success_cnt_31"].apply(lambda x: (x-t_mean24)/(t_max24-t_min24))
test_data["user_success_cnt_31"] = test_data["user_success_cnt_31"].apply(lambda x: (x-t_mean24)/(t_max24-t_min24))

t_max25 = np.max(train_data["user_success_cnt_90"])
t_mean25 = np.mean(train_data["user_success_cnt_90"])
t_min25 = np.min(train_data["user_success_cnt_90"])
train_data["user_success_cnt_90"] = train_data["user_success_cnt_90"].apply(lambda x: (x-t_mean25)/(t_max25-t_min25))
test_data["user_success_cnt_90"] = test_data["user_success_cnt_90"].apply(lambda x: (x-t_mean25)/(t_max25-t_min25))

t_max26 = np.max(train_data["user_success_amt_15"])
t_mean26 = np.mean(train_data["user_success_amt_15"])
t_min26 = np.min(train_data["user_success_amt_15"])
train_data["user_success_amt_15"] = train_data["user_success_amt_15"].apply(lambda x: (x-t_mean26)/(t_max26-t_min26))
test_data["user_success_amt_15"] = test_data["user_success_amt_15"].apply(lambda x: (x-t_mean26)/(t_max26-t_min26))

t_max27 = np.max(train_data["user_success_amt_31"])
t_mean27 = np.mean(train_data["user_success_amt_31"])
t_min27 = np.min(train_data["user_success_amt_31"])
train_data["user_success_amt_31"] = train_data["user_success_amt_31"].apply(lambda x: (x-t_mean27)/(t_max27-t_min27))
test_data["user_success_amt_31"] = test_data["user_success_amt_31"].apply(lambda x: (x-t_mean27)/(t_max27-t_min27))

t_max28 = np.max(train_data["user_success_amt_90"])
t_mean28 = np.mean(train_data["user_success_amt_90"])
t_min28 = np.min(train_data["user_success_amt_90"])
train_data["user_success_amt_90"] = train_data["user_success_amt_90"].apply(lambda x: (x-t_mean28)/(t_max28-t_min28))
test_data["user_success_amt_90"] = test_data["user_success_amt_90"].apply(lambda x: (x-t_mean28)/(t_max28-t_min28))

t_max29 = np.max(train_data["last_due_date_distance"])
t_mean29 = np.mean(train_data["last_due_date_distance"])
t_min29 = np.min(train_data["last_due_date_distance"])
train_data["last_due_date_distance"] = train_data["last_due_date_distance"].apply(lambda x: (x-t_mean29)/(t_max29-t_min29))
test_data["last_due_date_distance"] = test_data["last_due_date_distance"].apply(lambda x: (x-t_mean29)/(t_max29-t_min29))

t_max30 = np.max(train_data["first_due_date_distance"])
t_mean30 = np.mean(train_data["first_due_date_distance"])
t_min30 = np.min(train_data["first_due_date_distance"])
train_data["first_due_date_distance"] = train_data["first_due_date_distance"].apply(lambda x: (x-t_mean30)/(t_max30-t_min30))
test_data["first_due_date_distance"] = test_data["first_due_date_distance"].apply(lambda x: (x-t_mean30)/(t_max30-t_min30))

t_max31 = np.max(train_data["user_repay_before_15"])
t_mean31 = np.mean(train_data["user_repay_before_15"])
t_min31 = np.min(train_data["user_repay_before_15"])
train_data["user_repay_before_15"] = train_data["user_repay_before_15"].apply(lambda x: (x-t_mean31)/(t_max31-t_min31))
test_data["user_repay_before_15"] = test_data["user_repay_before_15"].apply(lambda x: (x-t_mean31)/(t_max31-t_min31))

t_max32 = np.max(train_data["user_repay_before_31"])
t_mean32 = np.mean(train_data["user_repay_before_31"])
t_min32 = np.min(train_data["user_repay_before_31"])
train_data["user_repay_before_31"] = train_data["user_repay_before_31"].apply(lambda x: (x-t_mean32)/(t_max32-t_min32))
test_data["user_repay_before_31"] = test_data["user_repay_before_31"].apply(lambda x: (x-t_mean32)/(t_max32-t_min32))

t_max33 = np.max(train_data["user_repay_before_90"])
t_mean33 = np.mean(train_data["user_repay_before_90"])
t_min33 = np.min(train_data["user_repay_before_90"])
train_data["user_repay_before_90"] = train_data["user_repay_before_90"].apply(lambda x: (x-t_mean33)/(t_max33-t_min33))
test_data["user_repay_before_90"] = test_data["user_repay_before_90"].apply(lambda x: (x-t_mean33)/(t_max33-t_min33))

t_max34 = np.max(train_data["user_repay_amt_before_15"])
t_mean34 = np.mean(train_data["user_repay_amt_before_15"])
t_min34 = np.min(train_data["user_repay_amt_before_15"])
train_data["user_repay_amt_before_15"] = train_data["user_repay_amt_before_15"].apply(lambda x: (x-t_mean34)/(t_max34-t_min34))
test_data["user_repay_amt_before_15"] = test_data["user_repay_amt_before_15"].apply(lambda x: (x-t_mean34)/(t_max34-t_min34))

t_max35 = np.max(train_data["user_repay_amt_before_31"])
t_mean35 = np.mean(train_data["user_repay_amt_before_31"])
t_min35 = np.min(train_data["user_repay_amt_before_31"])
train_data["user_repay_amt_before_31"] = train_data["user_repay_amt_before_31"].apply(lambda x: (x-t_mean35)/(t_max35-t_min35))
test_data["user_repay_amt_before_31"] = test_data["user_repay_amt_before_31"].apply(lambda x: (x-t_mean35)/(t_max35-t_min35))

t_max36 = np.max(train_data["user_repay_amt_before_90"])
t_mean36 = np.mean(train_data["user_repay_amt_before_90"])
t_min36 = np.min(train_data["user_repay_amt_before_90"])
train_data["user_repay_amt_before_90"] = train_data["user_repay_before_90"].apply(lambda x: (x-t_mean36)/(t_max36-t_min36))
test_data["user_repay_amt_before_90"] = test_data["user_repay_amt_before_90"].apply(lambda x: (x-t_mean36)/(t_max36-t_min36))

t_max37 = np.max(train_data["last_repay_date_distance"])
t_mean37 = np.mean(train_data["last_repay_date_distance"])
t_min37 = np.min(train_data["last_repay_date_distance"])
train_data["last_repay_date_distance"] = train_data["last_repay_date_distance"].apply(lambda x: (x-t_mean37)/(t_max37-t_min37))
test_data["last_repay_date_distance"] = test_data["last_repay_date_distance"].apply(lambda x: (x-t_mean37)/(t_max37-t_min37))

t_max38 = np.max(train_data["first_repay_date_distance"])
t_mean38 = np.mean(train_data["first_repay_date_distance"])
t_min38 = np.min(train_data["first_repay_date_distance"])
train_data["first_repay_date_distance"] = train_data["first_repay_date_distance"].apply(lambda x: (x-t_mean38)/(t_max38-t_min38))
test_data["first_repay_date_distance"] = test_data["first_repay_date_distance"].apply(lambda x: (x-t_mean38)/(t_max38-t_min38))

train_data.to_csv("../data/feature_dealing/dealing_v4_train.csv", index=False)
test_data.to_csv("../data/feature_dealing/dealing_v4_test.csv", index=False)
