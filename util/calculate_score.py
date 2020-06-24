# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: calculate_score.py
    @time: 2020/4/20 20:27
    
    @introduce: Just a __init__.py file
"""
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score
import pandas as pd
import numpy as np


def load_data():

    print("\nLoad data...")

    ppd_train = pd.read_csv("data/init/credit_train.csv")
    ppd_test = pd.read_csv("data/init/credit_test.csv")
    x_train_pd = ppd_train.drop(columns=["Class"])
    y_train_pd = ppd_train["Class"]
    x_train_np = x_train_pd.values
    y_train_np = y_train_pd.values

    x_test_pd = ppd_test.drop(columns=["Class"])
    y_test_pd = ppd_test["Class"]
    x_test_np = x_test_pd.values
    y_test_np = y_test_pd.values
    """
    ppd_train = pd.read_csv("data/feature_dealing/dealing_v4_train.csv")
    ppd_test = pd.read_csv("data/feature_dealing/dealing_v4_test.csv")
    x_train_pd = ppd_train.drop(columns=["label", "user_id", "listing_id", "auditing_date", "due_date", "month_rate",
                                         "day_of_week", "week_of_month", "day_of_month", "month_of_year",
                                         "user_do_0_3", "user_do_3_6", "user_do_6_9", "user_do_9_12",
                                         "user_do_12_15", "user_do_15_18", "user_do_18_21", "user_do_21_24",
                                         "user_do_0_3_rate", "user_do_3_6_rate", "user_do_6_9_rate", "user_do_9_12_rate",
                                         "user_do_12_15_rate", "user_do_15_18_rate", "user_do_18_21_rate",
                                         "user_do_21_24_rate"])
    y_train_pd = ppd_train["label"]
    x_train_np = x_train_pd.values
    y_train_np = y_train_pd.values

    x_test_pd = ppd_test.drop(columns=["label", "user_id", "listing_id", "auditing_date", "due_date", "month_rate",
                                       "day_of_week", "week_of_month", "day_of_month", "month_of_year",
                                       "user_do_0_3", "user_do_3_6", "user_do_6_9", "user_do_9_12",
                                       "user_do_12_15", "user_do_15_18", "user_do_18_21", "user_do_21_24",
                                       "user_do_0_3_rate", "user_do_3_6_rate", "user_do_6_9_rate", "user_do_9_12_rate",
                                       "user_do_12_15_rate", "user_do_15_18_rate", "user_do_18_21_rate",
                                       "user_do_21_24_rate"])
    y_test_pd = ppd_test["label"]
    x_test_np = x_test_pd.values
    y_test_np = y_test_pd.values
    """
    return x_train_np, x_test_np, y_train_np, y_test_np


def auc_prc(label, y_predict):
    return average_precision_score(label, y_predict)


def f1_optim(label, y_predict):
    y_predict = y_predict.copy()
    prec, reca, _ = precision_recall_curve(label, y_predict)
    f1 = 2 * (prec * reca) / (prec + reca)
    return max(f1)


def gm_optim(label, y_predict):
    y_predict = y_predict.copy()
    prec, reca, _ = precision_recall_curve(label, y_predict)
    gm = np.power((prec * reca), 0.5)
    return max(gm)


def pre_optim(label, y_predict):
    y_predict = y_predict.copy()
    for i in range(len(y_predict)):
        if y_predict[i] >= 0.5:
            y_predict[i] = 1
        else:
            y_predict[i] = 0
    p = precision_score(label, y_predict)
    return p


def rec_optim(label, y_predict):
    y_predict = y_predict.copy()
    for i in range(len(y_predict)):
        if y_predict[i] >= 0.5:
            y_predict[i] = 1
        else:
            y_predict[i] = 0
    r = recall_score(label, y_predict)
    return r


def load_ppd_data(name):

    str_name = "data/demo/" + name
    ppd_test = pd.read_csv(str_name)
    x_test_pd = ppd_test.drop(columns=["label", "user_id", "listing_id", "auditing_date", "due_date", "month_rate",
                                       "day_of_week", "week_of_month", "day_of_month", "month_of_year",
                                       "user_do_0_3", "user_do_3_6", "user_do_6_9", "user_do_9_12",
                                       "user_do_12_15", "user_do_15_18", "user_do_18_21", "user_do_21_24",
                                       "user_do_0_3_rate", "user_do_3_6_rate", "user_do_6_9_rate", "user_do_9_12_rate",
                                       "user_do_12_15_rate", "user_do_15_18_rate", "user_do_18_21_rate",
                                       "user_do_21_24_rate"])
    y_test_pd = ppd_test["label"]
    x_test_np = x_test_pd.values
    y_test_np = y_test_pd.values

    return x_test_np, y_test_np


def load_credit_data(name):

    str_name = "data/demo/" + name
    credit_test = pd.read_csv(str_name)

    x_test_pd = credit_test.drop(columns=["Class"])
    y_test_pd = credit_test["Class"]
    x_test_np = x_test_pd.values
    y_test_np = y_test_pd.values

    return x_test_np, y_test_np


def predict_new(x, models):
    # 根据以前的模型对数据属于多数类和少数类投票
    y_predict = np.array([model.predict_proba(x) for model in models]).mean(axis=0)
    if y_predict.ndim == 1:
        y_predict = y_predict[:, np.newaxis]
    if y_predict.shape[1] == 1:
        y_predict = np.append(1 - y_predict, y_predict, axis=1)
    return y_predict
