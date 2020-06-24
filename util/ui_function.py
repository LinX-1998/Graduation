# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: ui_function.py
    @time: 2020/5/14 13:01
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from util.calculate_score import *
from sklearn.externals import joblib

# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
# 设置pandas显示文件全部列
pd.set_option("display.max_columns", None)

demo_data_name = ["拍拍贷全量数据", "信用卡全量数据"]
demo_data_change = ["ppd_all.csv", "credit_all.csv"]


def check(name):
    for i in range(len(demo_data_name)):
        if name == demo_data_name[i]:
            return demo_data_change[i]


def check_model(pos, name):
    models = []
    if pos == "ppd":
        if name == "SPE+LR":
            models.append([0.215, 0.254, 0.333])
        elif name == "SPE+DT":
            models.append([0.181, 0.265, 0.336])
        elif name == "SPE+MLP":
            models.append([0.209, 0.295, 0.343])
        elif name == "SPE+RF":
            models.append([0.257, 0.323, 0.353])
        elif name == "RUS+LR":
            models.append([0.159, 0.292, 0.333])
        elif name == "RUS+DT":
            models.append([0.123, 0.214, 0.333])
        elif name == "RUS+MLP":
            models.append([0.154, 0.281, 0.334])
        elif name == "RUS+RF":
            models.append([0.163, 0.299, 0.337])
        elif name == "SMOTE+LR":
            models.append([0.163, 0.301, 0.333])
        elif name == "SMOTE+DT":
            models.append([0.122, 0.200, 0.333])
        elif name == "SMOTE+MLP":
            models.append([0.122, 0.202, 0.333])
        elif name == "SMOTE+RF":
            models.append([0.134, 0.199, 0.333])
        elif name == "SMOTEENN+LR":
            models.append([0.152, 0.276, 0.333])
        elif name == "SMOTEENN+DT":
            models.append([0.126, 0.207, 0.333])
        elif name == "SMOTEENN+MLP":
            models.append([0.124, 0.207, 0.333])
        else:
            models.append([0.157, 0.234, 0.333])
        return models
    else:
        if name == "SPE+LR":
            models.append([0.806, 0.823, 0.833])
        elif name == "SPE+DT":
            models.append([0.747, 0.788, 0.795])
        elif name == "SPE+MLP":
            models.append([0.550, 0.612, 0.626])
        elif name == "SPE+RF":
            models.append([0.818, 0.857, 0.863])
        elif name == "RUS+LR":
            models.append([0.089, 0.187, 0.297])
        elif name == "RUS+DT":
            models.append([0.009, 0.020, 0.092])
        elif name == "RUS+MLP":
            models.append([0.002, 0.005, 0.040])
        elif name == "RUS+RF":
            models.append([0.044, 0.094, 0.207])
        elif name == "SMOTE+LR":
            models.append([0.258, 0.447, 0.504])
        elif name == "SMOTE+DT":
            models.append([0.024, 0.081, 0.086])
        elif name == "SMOTE+MLP":
            models.append([0.123, 0.238, 0.337])
        elif name == "SMOTE+RF":
            models.append([0.685, 0.825, 0.827])
        elif name == "SMOTEENN+LR":
            models.append([0.267, 0.457, 0.513])
        elif name == "SMOTEENN+DT":
            models.append([0.035, 0.129, 0.135])
        elif name == "SMOTEENN+MLP":
            models.append([0.115, 0.217, 0.304])
        else:
            models.append([0.697, 0.832, 0.835])
        return models


def model_check(combobox_model_1, combobox_model_2, listbox, label_result_model_1, label_result_model_2,
                  label_result_1, label_result_2, label_result_3, label_result_4, label_result_5,
                  label_result_6, label_result_7):
    model_1_name = combobox_model_1.get()
    model_2_name = combobox_model_2.get()
    data_name = ""
    models = []
    if model_1_name != "--请选择--" and model_2_name == "--请选择--":
        try:
            data_name = listbox.get(listbox.curselection())
        except:
            listbox.activate(0)
            listbox.selection_set(0)
            data_name = listbox.get(0)
        str_name = check(data_name)
        if str_name == "ppd_all.csv":
            models = check_model("ppd", model_1_name)
        else:
            models = check_model("card", model_1_name)
        label_result_model_1["text"] = model_1_name + "模型"
        label_result_1["text"] = "PRC值: {:.3f}".format(models[0][0])
        label_result_2["text"] = "F1值: {:.3f}".format(models[0][1])
        label_result_3["text"] = "G-mean值: {:.3f}".format(models[0][2])
    else:
        label_result_model_1["text"] = "错误的配置"


def model_compare(combobox_model_1, combobox_model_2, listbox, label_result_model_1, label_result_model_2,
                  label_result_1, label_result_2, label_result_3, label_result_4, label_result_5,
                  label_result_6, label_result_7):
    model_1_name = combobox_model_1.get()
    model_2_name = combobox_model_2.get()
    models1 = []
    models2 = []
    data_name = ""
    if model_1_name != "--请选择--" and model_2_name != "--请选择--":
        try:
            data_name = listbox.get(listbox.curselection())
        except:
            listbox.activate(0)
            listbox.selection_set(0)
            data_name = listbox.get(0)
        str_name = check(data_name)
        if str_name == "ppd_all.csv":
            models1 = check_model("ppd", model_1_name)
            models2 = check_model("ppd", model_2_name)
        else:
            models1 = check_model("card", model_1_name)
            models2 = check_model("card", model_2_name)
        label_result_model_1["text"] = model_1_name + "模型"
        label_result_model_2["text"] = model_2_name + "模型"
        label_result_1["text"] = "PRC值: {:.3f}".format(models1[0][0])
        label_result_2["text"] = "F1值: {:.3f}".format(models1[0][1])
        label_result_3["text"] = "G-mean值: {:.3f}".format(models1[0][2])
        label_result_4["text"] = "vs".center(40, "-")
        label_result_5["text"] = "PRC值: {:.3f}".format(models2[0][0])
        label_result_6["text"] = "F1值: {:.3f}".format(models2[0][1])
        label_result_7["text"] = "G-mean值: {:.3f}".format(models2[0][2])
    else:
        label_result_model_1["text"] = "错误的配置"


def new_start(combobox_model_1, combobox_model_2, listbox, label_result_model_1, label_result_model_2,
              label_result_1, label_result_2, label_result_3, label_result_4, label_result_5,
              label_result_6, label_result_7):
    combobox_model_1.set("--请选择--")
    combobox_model_2.set("--请选择--")
    listbox.activate(0)
    try:
        listbox.selection_clear(listbox.curselection())
    except:
        pass
    listbox.selection_set(0)
    label_result_model_1["text"] = ""
    label_result_model_2["text"] = ""
    label_result_1["text"] = ""
    label_result_2["text"] = ""
    label_result_3["text"] = ""
    label_result_4["text"] = ""
    label_result_5["text"] = ""
    label_result_6["text"] = ""
    label_result_7["text"] = ""
