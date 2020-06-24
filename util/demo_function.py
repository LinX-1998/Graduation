# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: demo_function.py
    @time: 2020/5/19 17:18
    
    @introduce: Just a __init__.py file
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from util.calculate_score import *
from sklearn.externals import joblib
from datetime import datetime

# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
# 设置pandas显示文件全部列
pd.set_option("display.max_columns", None)

demo_data_name = ["拍拍贷全量数据", "信用卡全量数据", "拍拍贷测试数据-1", "拍拍贷测试数据-2", "拍拍贷测试数据-3",
                  "信用卡测试数据-1", "信用卡测试数据-2", "信用卡测试数据-3"]
demo_data_change = ["ppd_all.csv", "credit_all.csv", "ppd_default_1.csv", "ppd_default_2.csv", "ppd_default_3.csv",
                    "credit_default_1.csv", "credit_default_2.csv", "credit_default_3.csv"]


def load_ppd_spe_model(cnt, name):
    models = []
    name1 = name[0:3]
    name2 = name[4:]
    name3 = name1 + "_" + name2 + "_" + str(cnt) + "_"
    for i in range(15):
        new_name = "model/ppd/" + name3 + str(i+1) + "_model.pkl"
        model = joblib.load(new_name)
        models.append(model)
    return models


def load_ppd_model(cnt, name):
    models = []
    num = 0
    for i in range(len(name)):
        if name[i] == "+":
            num = i
    name1 = name[0:num]
    name2 = name[num+1:]
    name3 = "model/ppd/" + name1 + "_" + name2 + "_" + str(cnt) + "_model.pkl"
    model = joblib.load(name3)
    models.append(model)
    return models


def load_card_spe_model(cnt, name):
    models = []
    name1 = name[0:3]
    name2 = name[4:]
    name3 = name1 + "_" + name2 + "_" + str(cnt) + "_"
    for i in range(15):
        new_name = "model/card/" + name3 + str(i+1) + "_model.pkl"
        model = joblib.load(new_name)
        models.append(model)
    return models


def load_card_model(cnt, name):
    models = []
    num = 0
    for i in range(len(name)):
        if name[i] == "+":
            num = i
    name1 = name[0:num]
    name2 = name[num + 1:]
    name3 = "model/card/" + name1 + "_" + name2 + "_" + str(cnt) + "_model.pkl"
    model = joblib.load(name3)
    models.append(model)
    return models


def check(name):
    for i in range(len(demo_data_name)):
        if name == demo_data_name[i]:
            return demo_data_change[i]


def check_model(pos, name):
    models = []
    if pos == "ppd":
        if name == "SPE+DT":
            for i in range(10):
                model = load_ppd_spe_model(i, name)
                models.append(model)
        elif name == "RUS+DT":
            for i in range(10):
                model = load_ppd_model(i, name)
                models.append(model)
        elif name == "SMOTE+DT":
            for i in range(10):
                model = load_ppd_model(i, name)
                models.append(model)
        elif name == "SMOTEENN+DT":
            for i in range(10):
                model = load_ppd_model(i, name)
                models.append(model)
        else:
            pass
        return models
    else:
        if name == "SPE+DT":
            for i in range(10):
                model = load_card_spe_model(i, name)
                models.append(model)
        elif name == "RUS+DT":
            for i in range(10):
                model = load_card_model(i, name)
                models.append(model)
        elif name == "SMOTE+DT":
            for i in range(10):
                model = load_card_model(i, name)
                models.append(model)
        elif name == "SMOTEENN+DT":
            for i in range(10):
                model = load_card_model(i, name)
                models.append(model)
        else:
            pass
        return models


def save_predict(name, y_predict, str_name):
    ans = []
    for i in y_predict:
        if i >= 0.5:
            ans.append(1)
        else:
            ans.append(0)
    if name == "ppd":
        new_name = "data/demo/" + str_name
        ppd_all = pd.read_csv(new_name)
        user = list(ppd_all["user_id"].values)
        listing = list(ppd_all["listing_id"].values)
        pd_ans = pd.DataFrame({"user_id": user, "listing_id": listing, "predict": ans})
        date = datetime.now()
        save_name = str(date.year) + str(date.month) + str(date.day) + str(date.hour) + str(date.minute) + str(date.second) + "_ppd.csv"
        pd_ans.to_csv(save_name, index=False)
    else:
        pd_ans = pd.DataFrame({"predict": ans})
        date = datetime.now()
        save_name = str(date.year) + str(date.month) + str(date.day) + str(date.hour) + str(date.minute) + str(date.second) + "_credit.csv"
        pd_ans.to_csv(save_name, index=False)


def model_check(combobox_model_1, combobox_model_2, listbox, label_result_model_1, label_result_model_2,
                  label_result_1, label_result_2, label_result_3, label_result_4, label_result_5,
                  label_result_6, label_result_7):
    model_1_name = combobox_model_1.get()
    model_2_name = combobox_model_2.get()
    model_name = ""
    data_name = ""
    models = []
    scores = []
    if (model_1_name != "--请选择--" and model_2_name == "--请选择--") or (model_1_name == "--请选择--" and model_2_name != "--请选择--"):
        if model_1_name == "请选择":
            model_name = model_2_name
        else:
            model_name = model_1_name
        try:
            data_name = listbox.get(listbox.curselection())
        except:
            listbox.activate(0)
            listbox.selection_set(0)
            data_name = listbox.get(0)
        str_name = check(data_name)
        if str_name[0:3] == "ppd":
            models = check_model("ppd", model_name)
            x_test, y_test = load_ppd_data(str_name)
            for i in range(10):
                y_predict = predict_new(x_test, models[i])[:, 1]
                if i == 0:
                    save_predict("ppd", y_predict, str_name)
                scores.append([
                    auc_prc(y_test, y_predict),
                    rec_optim(y_test, y_predict),
                    f1_optim(y_test, y_predict),
                    gm_optim(y_test, y_predict)
                ])
        else:
            models = check_model("card", model_name)
            x_test, y_test = load_credit_data(str_name)
            for i in range(10):
                y_predict = predict_new(x_test, models[i])[:, 1]
                if i == 0:
                    save_predict("card", y_predict, str_name)
                scores.append([
                    auc_prc(y_test, y_predict),
                    rec_optim(y_test, y_predict),
                    f1_optim(y_test, y_predict),
                    gm_optim(y_test, y_predict)
                ])
        data_scores = pd.DataFrame(scores, columns=["PRCurve", "Recall", "F1", "G-mean"])
        label_result_model_1["text"] = model_1_name + "模型"
        label_result_1["text"] = "PRC值: {:.3f}".format(data_scores["PRCurve"].mean())
        label_result_2["text"] = "Recall值: {:.3f}".format(data_scores["Recall"].mean())
        label_result_3["text"] = "F1值: {:.3f}".format(data_scores["F1"].mean())
        label_result_4["text"] = "G-mean值: {:.3f}".format(data_scores["G-mean"].mean())
        label_result_model_2["text"] = ""
        label_result_5["text"] = ""
        label_result_6["text"] = ""
        label_result_7["text"] = ""
    else:
        label_result_model_1["text"] = "错误的配置"
        label_result_model_2["text"] = ""
        label_result_1["text"] = ""
        label_result_2["text"] = ""
        label_result_3["text"] = ""
        label_result_4["text"] = ""
        label_result_5["text"] = ""
        label_result_6["text"] = ""
        label_result_7["text"] = ""


def model_compare(combobox_model_1, combobox_model_2, listbox, label_result_model_1, label_result_model_2,
                  label_result_1, label_result_2, label_result_3, label_result_4, label_result_5,
                  label_result_6, label_result_7):
    model_1_name = combobox_model_1.get()
    model_2_name = combobox_model_2.get()
    models1 = []
    models2 = []
    data_name = ""
    scores1 = []
    scores2 = []
    if model_1_name != "--请选择--" and model_2_name != "--请选择--":
        try:
            data_name = listbox.get(listbox.curselection())
        except:
            listbox.activate(0)
            listbox.selection_set(0)
            data_name = listbox.get(0)
        str_name = check(data_name)
        if str_name[0:3] == "ppd":
            models1 = check_model("ppd", model_1_name)
            models2 = check_model("ppd", model_2_name)
            x_test, y_test = load_ppd_data(str_name)
            for i in range(10):
                y_predict1 = predict_new(x_test, models1[i])[:, 1]
                y_predict2 = predict_new(x_test, models2[i])[:, 1]
                scores1.append([
                    auc_prc(y_test, y_predict1),
                    f1_optim(y_test, y_predict1),
                    gm_optim(y_test, y_predict1)
                ])
                scores2.append([
                    auc_prc(y_test, y_predict2),
                    f1_optim(y_test, y_predict2),
                    gm_optim(y_test, y_predict2)
                ])
        else:
            models1 = check_model("card", model_1_name)
            models2 = check_model("card", model_2_name)
            x_test, y_test = load_credit_data(str_name)
            for i in range(10):
                y_predict1 = predict_new(x_test, models1[i])[:, 1]
                y_predict2 = predict_new(x_test, models2[i])[:, 1]
                scores1.append([
                    auc_prc(y_test, y_predict1),
                    f1_optim(y_test, y_predict1),
                    gm_optim(y_test, y_predict1)
                ])
                scores2.append([
                    auc_prc(y_test, y_predict2),
                    f1_optim(y_test, y_predict2),
                    gm_optim(y_test, y_predict2)
                ])
        data_scores1 = pd.DataFrame(scores1, columns=["PRCurve", "F1", "G-mean"])
        data_scores2 = pd.DataFrame(scores2, columns=["PRCurve", "F1", "G-mean"])
        label_result_model_1["text"] = model_1_name + "模型"
        label_result_1["text"] = "PRC值: {:.3f}".format(data_scores1["PRCurve"].mean())
        label_result_2["text"] = "F1值: {:.3f}".format(data_scores1["F1"].mean())
        label_result_3["text"] = "G-mean值: {:.3f}".format(data_scores1["G-mean"].mean())
        label_result_4["text"] = "vs".center(40, "-")
        label_result_model_2["text"] = model_2_name + "模型"
        label_result_5["text"] = "PRC值: {:.3f}".format(data_scores2["PRCurve"].mean())
        label_result_6["text"] = "F1值: {:.3f}".format(data_scores2["F1"].mean())
        label_result_7["text"] = "G-mean值: {:.3f}".format(data_scores2["G-mean"].mean())
    else:
        label_result_model_1["text"] = "错误的配置"
        label_result_model_2["text"] = ""
        label_result_1["text"] = ""
        label_result_2["text"] = ""
        label_result_3["text"] = ""
        label_result_4["text"] = ""
        label_result_5["text"] = ""
        label_result_6["text"] = ""
        label_result_7["text"] = ""


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
