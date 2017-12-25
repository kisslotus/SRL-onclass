# -*- coding:utf-8 -*-

# 格式均为：A/B/C，A为词，B为POS标签，C为SRL标签
# 初始文件路径
train_path = "data/cpbtrain.txt"
test_path = "data/cpbtest.txt"
dev_path = "data/cpbdev.txt"
pre_word_path = "data/vocab/word_vectors_pre.txt"

def to_index(sequences):
    """
    to_categorial函数的反函数，把one-hot转回idx
    """
    res = []
    for seq in sequences:
        for i, val in enumerate(seq):
            if val==1:
                res.append(i)
                break
    return res

def max_word():
    """
    统计所有数据中最长句子的单词数量
    """
    res = 0
    # 训练集
    sents = open(train_path, "r").readlines()
    tri_words = [sent.split() for sent in sents]  # 每个元素是A/B/C
    for sent in tri_words:
        res = max(res,len(sent))
    # 开发集
    sents = open(dev_path, "r").readlines()
    tri_words = [sent.split() for sent in sents]  # 每个元素是A/B/C
    for sent in tri_words:
        res = max(res, len(sent))
    # 测试集
    sents = open(test_path, "r").readlines()
    tri_words = [sent.split() for sent in sents]  # 每个元素是A/B/C
    for sent in tri_words:
        res = max(res, len(sent))
    return res