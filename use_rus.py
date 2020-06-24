# -*- coding: utf-8 -*-
"""
    @Author: LinXiang
    @Email: linxiang-1998@outlook.com
    @file: use_rus.py
    @time: 2020/4/20 20:54
    
    @introduce: Just a __init__.py file
"""
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sampling_train.RandomUnderSampling import RandomUnderSampling
from util.calculate_score import *
import argparse
from time import clock
from tqdm import trange
import warnings

warnings.filterwarnings("ignore")

# 设置采样方法和种子，防止多次测试结果不同
METHODS = ["SPE", "RUS", "SMOTE", "SMOTEENN"]
RANDOM_STATE = 42


def parse():
    parser = argparse.ArgumentParser(
        description="SelfPacedEnsemble",
        usage="use_spe.py --method <method> --n_estimators <integer> --runs <integer>"
    )
    parser.add_argument("--method", type=str, default="RUS", help="Sampling methods")
    parser.add_argument("--n_estimators", type=int, default=10, help="Number of base estimators")
    parser.add_argument("--runs", type=int, default=10, help="Number of independent runs")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse()
    n_estimators = args.n_estimators
    runs = args.runs
    method = args.method

    # Load train/test data
    x_train, x_test, y_train, y_test = load_data()

    # Train
    print("\nRunning method:\t\t{} - {} estimators in {} independent run(s) ...".format(method, n_estimators, runs))
    scores = []
    times = []
    try:
        with trange(runs) as t:
            for _ in t:
                if method == "SPE":
                    pass
                elif method == "RUS":
                    model = RandomUnderSampling(
                        cnt=_,
                        base_estimator=LogisticRegression(),
                        model_name="LR")
                elif method == "SMOTE":
                    pass
                else:
                    pass
                start_time = clock()
                model.fit(x_train, y_train)
                times.append(clock() - start_time)
                y_predict = model.predict_proba(x_test)[:, 1]
                scores.append([
                    auc_prc(y_test, y_predict),
                    pre_optim(y_test, y_predict),
                    rec_optim(y_test, y_predict),
                    f1_optim(y_test, y_predict),
                    gm_optim(y_test, y_predict)
                ])
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()

    print("\n------------------------------")
    print("Metrics:")
    data_scores = pd.DataFrame(scores, columns=["PRCurve", "Precision", "Recall", "F1", "G-mean"])
    data_scores.to_csv("check.csv")
    for metric in data_scores.columns.tolist():
        print("{}\tmean:{:.3f}  std:{:.3f}".format(metric, data_scores[metric].mean(), data_scores[metric].std()))
    return


if __name__ == "__main__":
    main()
