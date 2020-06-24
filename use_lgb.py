# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: use_lgb.py
    @time: 2020/4/28 6:21
    
    @introduce: Just a __init__.py file
"""
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
# 设置pandas显示文件全部列
pd.set_option("display.max_columns", None)

# 加载你的数据
print("Load data...")
train_data = pd.read_csv("data/feature_dealing/dealing_v4_train.csv")
test_data = pd.read_csv("data/feature_dealing/dealing_v4_test.csv")
x_train = train_data.drop(columns=["label", "user_id", "listing_id", "auditing_date", "due_date", "gender",
                                   "day_of_week", "week_of_month", "day_of_month", "month_of_year",
                                   "user_do_0_3", "user_do_3_6", "user_do_6_9", "user_do_9_12",
                                   "user_do_12_15", "user_do_15_18", "user_do_18_21", "user_do_21_24",
                                   "is_age_10", "is_age_20", "is_age_30", "is_age_40", "is_age_50",
                                   "is_age_60", "is_age_70", "is_age_80", "is_age_90", "is_age_100",
                                   "user_repay_before_31", "user_do_3_6_rate", "rate","user_deadline_amt_15",
                                   "user_success_amt_15", "user_repay_before_15", "user_default_cnt_31",
                                   "user_repay_amt_before_15", "user_default_amt_15", "user_deadline_cnt_15",
                                   "user_default_cnt_15", "user_default_amt_90", "user_deadline_cnt_31"])
y_train = train_data["label"]
x_test = test_data.drop(columns=["label", "user_id", "listing_id", "auditing_date", "due_date", "gender",
                                 "day_of_week", "week_of_month", "day_of_month", "month_of_year",
                                 "user_do_0_3", "user_do_3_6", "user_do_6_9", "user_do_9_12",
                                 "user_do_12_15", "user_do_15_18", "user_do_18_21", "user_do_21_24",
                                 "is_age_10", "is_age_20", "is_age_30", "is_age_40", "is_age_50",
                                 "is_age_60", "is_age_70", "is_age_80", "is_age_90", "is_age_100",
                                 "user_repay_before_31", "user_do_3_6_rate", "rate", "user_deadline_amt_15",
                                 "user_success_amt_15", "user_repay_before_15", "user_default_cnt_31",
                                 "user_repay_amt_before_15", "user_default_amt_15", "user_deadline_cnt_15",
                                 "user_default_cnt_15", "user_default_amt_90", "user_deadline_cnt_31"])
y_test = test_data["label"]

# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# 将参数写成字典下形式
params = {
    "task": "train",
    "boosting_type": "gbdt",  # 设置提升类型
    "objective": "binary",  # 目标函数
    "learning_rate": 0.03,  # 学习速率
    "feature_fraction": 0.9,  # 建树的特征选择比例
    "bagging_fraction": 0.9,  # 建树的样本采样比例
    "bagging_seed": 0,
    "bagging_freq": 1,  # k 意味着每 k 次迭代执行bagging
    "verbose": 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    "metric": ["binary_logloss"],  # 评估函数
    "num_leaves": 256,   # 叶子节点数
    "is_unbalance": True,
    "random_state": 2020
}

print("Start training...")
# 训练 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=30000, valid_sets=lgb_eval, early_stopping_rounds=100)

print("Save model...")
# 保存模型到文件
joblib.dump(gbm, "model/gbm.pkl")
lgb.plot_importance(gbm, max_num_features=30)
plt.title("Feature importance")
plt.show()

importance = gbm.feature_importance(importance_type="split")
feature_name = gbm.feature_name()
feature_importance = pd.DataFrame({"feature_name": feature_name, "importance": importance})
feature_importance = feature_importance.sort_values(by="importance", ascending=False)
feature_importance.reset_index()
feature_importance.to_csv("feature_importance3.csv", index=False)
