# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: use_demo.py
    @time: 2020/5/19 17:19
    
    @introduce: Just a __init__.py file
"""
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
from collections import Counter
from util.calculate_score import *
from util.demo_function import model_check, model_compare, new_start
import warnings


# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
# 设置pandas显示文件全部列
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

tk = Tk()
tk.title(u"数据逾期预测与比较")

menu_bar = Menu(tk, bg="grey")
file_menu = Menu(menu_bar, tearoff=False)
file_menu.add_command(label="退出", command=tk.quit)
menu_bar.add_cascade(label="文件", menu=file_menu)
tk.config(menu=menu_bar)

frame_model = Frame(width=400, height=120)
frame_data = Frame(width=400, height=300)
frame_button = Frame(width=400, height=30, bg="pink")
frame_result = Frame(width=218, height=456)

frame_model.grid(row=0, column=0, padx=1, pady=2)
frame_data.grid(row=1, column=0, padx=1, pady=1)
frame_button.grid(row=2, column=0, padx=1, pady=2)
frame_result.grid(row=0, column=1, rowspan=3, padx=1, pady=2)

frame_result.grid_propagate(0)

"""模型选择"""
label_model = Label(frame_model, text="模型选择")
label_model.grid(row=0, column=0, columnspan=2)

str_value_1 = StringVar()
label_model_1 = Label(frame_model, text="模型1选择:")
label_model_1.grid(row=1, column=0, pady=8)
combobox_model_1 = ttk.Combobox(frame_model, textvariable=str_value_1)
combobox_model_1["value"] = ("--请选择--",
                             "SPE+DT", "RUS+DT", "SMOTE+DT", "SMOTEENN+DT")
combobox_model_1.current(0)
combobox_model_1.grid(row=1, column=1)

str_value_2 = StringVar()
label_model_2 = Label(frame_model, text="模型2选择:")
label_model_2.grid(row=2, column=0)
combobox_model_2 = ttk.Combobox(frame_model, textvariable=str_value_2)
combobox_model_2["value"] = ("--请选择--",
                             "SPE+DT", "RUS+DT", "SMOTE+DT", "SMOTEENN+DT")
combobox_model_2.current(0)
combobox_model_2.grid(row=2, column=1)

"""数据选择"""
label_data = Label(frame_data, text="数据集选择")
label_data.grid(row=0, column=0, columnspan=2, pady=5)
sb = Scrollbar(frame_data)
sb.grid(row=1, column=1, sticky=NS)
list_value = StringVar()
list_value.set(("拍拍贷全量数据", "信用卡全量数据", "拍拍贷测试数据-1", "拍拍贷测试数据-2", "拍拍贷测试数据-3",
                "信用卡测试数据-1", "信用卡测试数据-2", "信用卡测试数据-3"))
listbox = Listbox(frame_data, width=55, height=15, listvariable=list_value, yscrollcommand=sb.set, selectbackground="white", selectforeground="red")
listbox.grid(row=1, column=0)

"""按钮选择"""
btn_check = Button(frame_button, text="预测", width=8, command=lambda: model_check(combobox_model_1, combobox_model_2, listbox,
                                                                              label_result_model_1, label_result_model_2,
                                                                              label_result_1, label_result_2, label_result_3,
                                                                              label_result_4, label_result_5, label_result_6,
                                                                              label_result_7))
btn_check.grid(row=0, column=0, padx=1)
btn_compare = Button(frame_button, text="比较", width=8, command=lambda: model_compare(combobox_model_1, combobox_model_2, listbox,
                                                                              label_result_model_1, label_result_model_2,
                                                                              label_result_1, label_result_2, label_result_3,
                                                                              label_result_4, label_result_5, label_result_6,
                                                                              label_result_7))
btn_compare.grid(row=0, column=1, padx=1)
btn_new = Button(frame_button, text="还原",  width=8, command=lambda: new_start(combobox_model_1, combobox_model_2, listbox,
                                                                              label_result_model_1, label_result_model_2,                                                                              label_result_1, label_result_2, label_result_3,
                                                                              label_result_4, label_result_5, label_result_6,
                                                                              label_result_7))
btn_new.grid(row=0, column=2, padx=1)

"""结果选择"""
label_result = Label(frame_result, text="结果显示", width=30)
label_result.grid(row=0, column=0, sticky=N, padx=1, pady=2)

label_result_model_1 = Label(frame_result, text="")
label_result_model_1.grid(row=1, column=0, pady=15)
label_result_1 = Label(frame_result, text="")
label_result_1.grid(row=2, column=0)
label_result_2 = Label(frame_result, text="")
label_result_2.grid(row=3, column=0, pady=15)
label_result_3 = Label(frame_result, text="")
label_result_3.grid(row=4, column=0)

label_result_4 = Label(frame_result, text="")
label_result_4.grid(row=5, column=0, pady=15)

label_result_model_2 = Label(frame_result, text="")
label_result_model_2.grid(row=6, column=0)
label_result_5 = Label(frame_result, text="")
label_result_5.grid(row=7, column=0, pady=15)
label_result_6 = Label(frame_result, text="")
label_result_6.grid(row=8, column=0)
label_result_7 = Label(frame_result, text="")
label_result_7.grid(row=9, column=0, pady=15)

tk.mainloop()
