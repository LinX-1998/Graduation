# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: SMOTE.py
    @time: 2020/4/20 21:11
    
    @introduce: Just a __init__.py file
"""
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.externals import joblib


class Smote():

    def __init__(self,
                 cnt=1,
                 base_estimator=DecisionTreeClassifier(),
                 model_name="DT",
                 random_state=None):
        self.cnt = cnt
        self.base_estimator_ = base_estimator
        self.estimators_ = []
        self.model_name = model_name
        self._random_state = random_state

    def _fit_base_estimator(self, x, y):
        return sklearn.base.clone(self.base_estimator_).fit(x, y)

    def fit(self, x, y):

        # 随机下采样
        smote = SMOTE()
        x_train, y_train = smote.fit_sample(x, y)
        self.estimators_.append(self._fit_base_estimator(x_train, y_train))

        for i in range(len(self.estimators_)):
            joblib.dump(self.estimators_[i], "model/card/SMOTE_"+self.model_name+"_"+str(self.cnt)+"_model.pkl")
        return self

    def predict_proba(self, x):
        # 根据以前的模型对数据属于多数类和少数类投票
        y_predict = np.array([model.predict(x) for model in self.estimators_]).mean(axis=0)
        if y_predict.ndim == 1:
            y_predict = y_predict[:, np.newaxis]
        if y_predict.shape[1] == 1:
            y_predict = np.append(1 - y_predict, y_predict, axis=1)
        return y_predict
