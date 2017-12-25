# -*- coding:utf-8 -*-

from Vocab import *
from model import SRLModel
from utils import *

# 构造Vocab对象
vocabs = get_vocabs()

# 构造模型
args = {
    'model_name' : "Final",
    'word_emb_dim' : 50,
    'preword_emb_dim' : 50,
    'pos_emb_dim' : 25,
    'role_nums' : vocabs['roles'].size,
    'bilstm_num' : 2,
    'word_vocab_size' : vocabs["words"].size,
    'max_words' : max_word(),
    'pos_vocab_size' : vocabs["pos"].size
}

model = SRLModel(**args)
model.build(dropout=0.5)

def train(summary_print=True, summary_pic=False, summary_fn = False, judgeTrain = True, judgeDev = True):
    # 构造输入
    # 构造训练集
    pre_train1 = np.load("data/pre_train-x1.npz")
    x_train = [pre_train1['word'], pre_train1['preword'], pre_train1['pos'], pre_train1['pred']]
    y_train = [np.load("data/pre_train-y.npy")]

    # 构造验证集
    pre_dev1 = np.load("data/pre_dev-x1.npz")
    x_dev = [pre_dev1['word'], pre_dev1['preword'], pre_dev1['pos'], pre_dev1['pred']]

    print u"输入处理完毕。。。"
    # 输出模型信息
    model.summary(printout=summary_print, pic=summary_pic, fn=summary_fn)
    # 训练
    model.train(x_train, y_train)
    # 评估模型
    if judgeTrain:
        model.judge(x_train, "train", train_path, vocabs)
    if judgeDev:
        model.judge(x_dev, "dev", dev_path, vocabs)

if __name__ == "__main__":
    train(summary_pic=True, summary_fn=True)
