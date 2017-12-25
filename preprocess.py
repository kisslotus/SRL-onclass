# -×- coding: utf-8 -×-

from Vocab import *
from utils import *
from keras.preprocessing.sequence import *
import os

def build_vocab_file(filepath):
    """
    预处理作业所给数据，为Vocab.py提供生成词典的源文件
    """
    sents = open(filepath, "r").readlines()
    tri_words = [sent.split() for sent in sents]  # 每个元素是A/B/C
    tri_words = [sent for sent in tri_words if len(sent) > 0]
    words = []
    for sent in tri_words:
        words.append([tri.split("/") for tri in sent])
    files = {typ:{} for typ in vocab_types}
    for sent in words:
        for word in sent:
            for i,typ in enumerate(vocab_types):
                if(word[i] not in files[typ]):
                    files[typ][word[i]] = 1
                else:
                    files[typ][word[i]] += 1

    # 写词典文件
    for typ in vocab_types:
        fn = get_vocab_path(typ)
        open(fn,"w").writelines(["%s %d\n"%(w,c) for w,c in files[typ].items()])

def get_xy(filepath):
    """
    对于训练集和开发集，返回x和y的初步分割结果
    """
    sents = open(filepath, "r").readlines()
    tri_words = [sent.split() for sent in sents]  # 每个元素是A/B/C
    tri_words = [sent for sent in tri_words if len(sent)>0]
    words = []
    for sent in tri_words:
        words.append([tri.split("/") for tri in sent])
    raws = [zip(*sent) for sent in words]
    # 找出谓词所在的地方
    for i,sent in enumerate(tri_words):
        pred = '<unk>'
        for j, word in enumerate(sent):
            # 找到谓词
            if word.endswith("/rel"):
                pred = word[:word.find('/')]
        raws[i].append([pred]*len(sent))
    return zip(*raws)

def loadGloVe(filename,emb_size):
    """
    导入GloVe预训练词向量
    """
    vocab = []
    embd = []
    vocab.append('<zero>') #装载不认识的词
    embd.append([0]*emb_size) #这个emb_size可能需要指定
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    # print('Loaded GloVe!')
    file.close()
    res = {}
    for i, word in enumerate(vocab):
        res[word] = i
    return res,embd

def encode(sent, vocab, embed, max_word):

    res = []
    for word in sent:
        if word in vocab:
            res.append(vocab[word])
        else:
            res.append(vocab['<unk>'])
    res += [0]*(max_word-len(res))
    return [embed[x] for x in res]

def encodes(sents,vocab,embed, max_word):
    res = []
    for sent in sents:
        res.append(encode(sent,vocab, embed, max_word))
    return res

def pre_input(raw_path, typ, is_y):
    """
    预处理训练集/开发集/测试集的输入和输出，把它们封装成模型接受的输入和输出
    模型训练时不需要再次处理，读取文件即可
    由于内存有限，x和y分开处理
    """
    # 构造Vocab对象
    vocabs = get_vocabs()
    MAX_WORDS = max_word()  # 一句话里最大的单词数

    if typ == "test":
        (word_raw, pos_raw, pred_raw) = get_xy(raw_path)
        is_y = False
    else:
        (word_raw, pos_raw, role_raw, pred_raw) = get_xy(raw_path)

    if is_y:

        role_raw = [vocabs['roles'].encode_sequence(sent) for sent in role_raw]
        role_raw = pad_sequences(role_raw, maxlen=MAX_WORDS, padding='post')
        # print role_raw[0]
        role_raw = [np.array(sent).reshape(MAX_WORDS, 1) for sent in role_raw]
        np.save("data/pre_%s-y" % typ, np.array(role_raw))
    else:
        vocab, embd = loadGloVe(pre_word_path, 50)
        # vocab_size = len(vocab)
        # embedding_dim = len(embd[0])
        word_d = [vocabs['words'].encode_sequence(sent) for sent in word_raw]
        word_d = pad_sequences(word_d, maxlen=MAX_WORDS, padding='post')
        preword_raw = encodes(word_raw, vocab, embd, MAX_WORDS)
        pos_raw = [vocabs['pos'].encode_sequence(sent) for sent in pos_raw]
        pos_raw = pad_sequences(pos_raw, maxlen=MAX_WORDS, padding='post')
        pred_raw = [vocabs['words'].encode_sequence(sent) for sent in pred_raw]
        pred_raw = pad_sequences(pred_raw, maxlen=MAX_WORDS, padding='post')
        np.savez("data/pre_%s-x1" % typ, word=np.array(word_d), pos=np.array(pos_raw), pred=np.array(pred_raw), preword=preword_raw)

def buildword(filepath):
    """
    生成给GloVe预训练模型的输入，目前已生成对应词向量文件，不需要再使用这个函数
    """
    sents = open(filepath, "r").readlines()
    tri_words = [sent.split() for sent in sents]  # 每个元素是A/B/C
    tri_words = [sent for sent in tri_words if len(sent) > 0]
    words = []
    for sent in tri_words:
        words.append([tri.split("/") for tri in sent])
    for i,_ in enumerate(words):
        words[i] = zip(*words[i])
    words = zip(*words)
    use = words[0]
    use = [' '.join(sent)+'\n' for sent in use]
    open("data/vocab/temp.txt","w").writelines(use)


if __name__ == '__main__':
    if (not os.path.exists("data/vocab/vocab_pos.txt"))  or (not os.path.exists("data/vocab/vocab_words.txt")) or (not os.path.exists("data/vocab/vocab_roles.txt")):
    #     print "lkj"
        print u"生成词典文件。。。"
        build_vocab_file(train_path)
        print u"文件生成完毕。。。"
    # 生成训练集输入输出
    if not os.path.exists("data/pre_train-x1.npz"):
        print u"生成训练集输入文件。。。"
        pre_input(train_path,"train",False)
        print u"文件生成完毕。。。"
    if not os.path.exists("data/pre_train-y.npy"):
        print u"生成训练集输出文件。。。"
        pre_input(train_path,"train",True)
        print u"文件生成完毕。。。"
    # 生成开发集输入输出
    if not os.path.exists("data/pre_dev-x1.npz"):
        print u"生成开发集输入文件。。。"
        pre_input(dev_path,"dev",False)
        print u"文件生成完毕。。。"
    if not os.path.exists("data/pre_dev-y.npy"):
        print u"生成开发集输出文件。。。"
        pre_input(dev_path,"dev",True)
        print u"文件生成完毕。。。"
    # 生成测试集输入
    if not os.path.exists("data/pre_test-x1.npz"):
        print u"生成测试集输入文件。。。"
        pre_input(test_path,"test",False)
        print u"文件生成完毕。。。"
