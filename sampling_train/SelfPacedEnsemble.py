# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: SelfPacedEnsemble.py
    @time: 2020/4/20 18:10
    
    @introduce: Just a __init__.py file
"""
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt

# 设置图像文字字体以及设置图像文字为utf-8
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False


class SelfPacedEnsemble():

    def __init__(self,
                 cnt=1,
                 base_estimator=DecisionTreeClassifier(),
                 hardness_function=lambda y_true, y_predict: np.absolute(y_true - y_predict),
                 n_estimators=10,
                 k_bins=10,
                 model_name="DT",
                 random_state=None):
        self.cnt = cnt
        self.base_estimator_ = base_estimator
        self.estimators_ = []
        self._hardness_function = hardness_function
        self._n_estimators = n_estimators
        self._k_bins = k_bins
        self.model_name = model_name
        self._random_state = random_state

    def _fit_base_estimator(self, x, y):
        return sklearn.base.clone(self.base_estimator_).fit(x, y)

    def _random_under_sampling(self, x_majority, y_majority, x_minority, y_minority):
        np.random.seed(self._random_state)
        idx = np.random.choice(len(x_majority), len(x_minority), replace=False)
        x_train = np.concatenate([x_majority[idx], x_minority])
        y_train = np.concatenate([y_majority[idx], y_minority])
        return x_train, y_train

    def _self_paced_under_sampling(self, x_majority, y_majority, x_minority, y_minority, i_estimator):
        # 确定预测结果和分类硬度
        y_predict_maj = self.predict_proba(x_majority)[:, 1]
        hardness = self._hardness_function(y_majority, y_predict_maj)

        # 分类硬度相同则进行随机采样
        if hardness.max() == hardness.min():
            x_train, y_train = self._random_under_sampling(x_majority, y_majority, x_minority, y_minority)
        else:
            # 准备进行分桶操作
            step = (hardness.max() - hardness.min()) / self._k_bins
            bins = []
            ave_contributions = []
            contributions = []
            for i_bins in range(self._k_bins):
                idx = ((hardness >= i_bins * step + hardness.min()) & (hardness < (i_bins + 1) * step + hardness.min()))
                if i_bins == (self._k_bins - 1):
                    idx = idx | (hardness == hardness.max())
                # 将多数类样本分到对应的桶里面
                bins.append(x_majority[idx])
                # 获取每个桶平均分类硬度
                ave_contributions.append(hardness[idx].mean())

            # 更新自定义步长参数α
            alpha = np.tan(np.pi * 0.5 * (i_estimator / (self._n_estimators - 1)))

            # 计算每个桶的采样权重
            weights = 1 / (ave_contributions + alpha)
            weights[np.isnan(weights)] = 0
            # 计算从每个桶中采样的个数
            n_sample_bins = len(x_minority) * weights / weights.sum()
            n_sample_bins = n_sample_bins.astype(int) + 1

            """
            ave_contributions = np.array(ave_contributions)
            ave_contributions[np.isnan(ave_contributions)] = 0.0
            for i in range(self._k_bins):
                contributions.append(n_sample_bins[i] * ave_contributions[i])
            fig = plt.figure()
            tt = plt.bar(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], contributions)
            plt.xlabel(u"桶编号")
            plt.ylabel(u"总贡献度")
            plt.title(u"桶编号-总贡献度柱状图")
            for b in tt:
                h = b.get_height()
                plt.text(b.get_x() + b.get_width() / 2, h, "%.2f" % float(h), ha="center", va="bottom")
            plt.show()
            """

            # 进行采样
            sampled_bins = []
            for i_bins in range(self._k_bins):
                # 如果桶中数目和采样数目最小值大于0，则需要采样
                if min(len(bins[i_bins]), n_sample_bins[i_bins]) > 0:
                    np.random.seed(self._random_state)
                    idx = np.random.choice(len(bins[i_bins]), min(len(bins[i_bins]), n_sample_bins[i_bins]), replace=False)
                    sampled_bins.append(bins[i_bins][idx])
            # 采样结束确定多数类样本
            x_train_majority = np.concatenate(sampled_bins, axis=0)
            y_train_majority = np.full(x_train_majority.shape[0], y_majority[0])
            x_train = np.concatenate([x_train_majority, x_minority])
            y_train = np.concatenate([y_train_majority, y_minority])
        return x_train, y_train

    def fit(self, x, y, label_majority=0, label_minority=1):
        self.estimators_ = []
        x_majority = x[y == label_majority]
        y_majority = y[y == label_majority]
        x_minority = x[y == label_minority]
        y_minority = y[y == label_minority]

        # 初始化下采样
        x_train, y_train = self._random_under_sampling(x_majority, y_majority, x_minority, y_minority)
        self.estimators_.append(self._fit_base_estimator(x_train, y_train))

        # 开始迭代
        for i_estimator in range(1, self._n_estimators):
            x_train, y_train = self._self_paced_under_sampling(x_majority, y_majority, x_minority, y_minority, i_estimator)
            self.estimators_.append(self._fit_base_estimator(x_train, y_train))

        for i in range(len(self.estimators_)):
            joblib.dump(self.estimators_[i], "model/card/SPE_"+self.model_name+"_" + str(self.cnt) + "_" + str(i+1)+"_model.pkl")

        return self

    def predict_proba(self, x):
        # 根据以前的模型对数据属于多数类和少数类投票
        y_predict = np.array([model.predict_proba(x) for model in self.estimators_]).mean(axis=0)
        if y_predict.ndim == 1:
            y_predict = y_predict[:, np.newaxis]
        if y_predict.shape[1] == 1:
            y_predict = np.append(1 - y_predict, y_predict, axis=1)
        return y_predict

    def predict(self, x):
        y_predict_binary = sklearn.preprocessing.binarize(
            self.predict_proba(x)[:, 1].reshape(1, -1), threshold=0.5)[0]
        return y_predict_binary
