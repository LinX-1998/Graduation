# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: calculate_date.py
    @time: 2020/4/20 15:21
    
    @introduce: Just a __init__.py file
"""
from datetime import datetime, timedelta


def calculate_two_day_difference(before_date, after_date):
    if type(before_date) != str or type(after_date) != str:
        return -1
    if before_date == r"\N" or after_date == r"\N":
        return -1
    before_date = datetime.strptime(before_date, "%Y-%m-%d")
    after_date = datetime.strptime(after_date, "%Y-%m-%d")
    return (after_date - before_date).days


def calculate_two_month_difference(before_month, after_month):
    if type(before_month) != str or type(after_month) != str:
        return -1
    if before_month == r"\N" or after_month == r"\N":
        return -1
    before_month_y = int(before_month[0:4])
    before_month_m = int(before_month[5:7])
    after_month_y = int(after_month[0:4])
    after_month_m = int(after_month[5:7])
    return (after_month_y - before_month_y) * 12 + (after_month_m - before_month_m)


def calculate_before_day(date, days):
    if type(date) != str or date == r"\N":
        return -1
    date = datetime.strptime(date, "%Y-%m-%d")
    before_date = date - timedelta(days=days)
    return str(before_date)[0:10]


def calculate_after_day(date, days):
    if type(date) != str or date == r"\N":
        return -1
    date = datetime.strptime(date, "%Y-%m-%d")
    after_date = date + timedelta(days=days)
    return str(after_date)[0:10]
