# -*- coding:utf-8 -*-

import argparse
from train import *

parser = argparse.ArgumentParser(
    description="Params for testing an SRL model")
parser.add_argument("--w",
                    help="Loading existed weights", type=str)
parser.add_argument("--t",
                    help="Test type: train, dev or test", type=str)

if __name__  == "__main__":
    args = parser.parse_args()  # 读取参数
    if args.w:
        # 如果有权重文件直接读
        model.load_model_weights(args.w)
    else:
        # 否则先训练
        train(summary_print=False, judgeDev=False, judgeTrain=False)
    if args.t == "train":
        # 构造训练集
        pre_train1 = np.load("data/pre_train-x1.npz")
        x_train = [pre_train1['word'], pre_train1['preword'], pre_train1['pos'], pre_train1['pred']]
        y_train = [np.load("data/pre_train-y.npy")]
        model.judge(x_train, "train", train_path, vocabs)
    elif args.t == "dev":
        # 构造验证集
        pre_dev1 = np.load("data/pre_dev-x1.npz")
        x_dev = [pre_dev1['word'], pre_dev1['preword'], pre_dev1['pos'], pre_dev1['pred']]

        model.judge(x_dev,"dev", dev_path, vocabs)
    else:
        # 读取测试集
        pre_test = np.load("data/pre_test-x1.npz")
        x_test = [pre_test['word'], pre_test['preword'], pre_test['pos'], pre_test['pred']]
        model.test(x_test, test_path, "test", vocabs)
