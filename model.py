# -*- coding:utf-8 -*-
from utils import *
import Vocab
import calc_f1 as J
import keras
import numpy as np
from keras.layers import *
from keras.models import *
from keras.preprocessing.sequence import *
from keras_contrib.layers import CRF
from keras.models import load_model
from keras.utils import plot_model
from keras.callbacks import *
from keras.constraints import Constraint
import keras.backend as K

class SRLModel():
    """
    # SRL模型
    模型结构如下：

    初始化：
    model_name：模型名字
    word_emb_dim：词向量维度
    preword_emb_dim：预训练词向量维度
    pos_emb_dim：POS向量维度
    role_nums：角色标签数量
    bilstm_num：BILSTM层数
    word_vocab_size：单词词典大小/谓词词典大小
    max_words：最长单词数
    pos_vocab_size:pos词典大小

    """
    def __init__(self, model_name, word_emb_dim, preword_emb_dim, pos_emb_dim,role_nums, bilstm_num, word_vocab_size, max_words, pos_vocab_size):
        # 导入多个config
        self.model_name = model_name
        self.word_emb_dim = word_emb_dim
        self.preword_emb_dim = preword_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.bilstm_num = bilstm_num
        self.role_nums = role_nums
        self.lstm_dim = role_nums*2
        self.max_words = max_words
        self.word_vocab_size = word_vocab_size
        self.pos_vocab_size = pos_vocab_size

    def build(self, dropout=0.5):
        """
        构造网络
        """
        # word_embedding部分，开启0屏蔽
        word_input = Input(shape=(self.max_words,), dtype='int32', name='word_input')
        word_emb = Embedding(self.word_vocab_size, self.word_emb_dim, input_length=self.max_words, name='word_emb',
                             mask_zero=True)(word_input)
        # 预训练的word_embedding
        pre_word_input = Input(shape=(self.max_words, self.preword_emb_dim), dtype='float32', name='pre_word_input')
        # pos_embedding部分
        pos_input = Input(shape=(self.max_words,), dtype='int32', name='pos_input')
        pos_emb = Embedding(self.pos_vocab_size, self.pos_emb_dim, input_length=self.max_words, name='pos_emb', mask_zero=True)(
            pos_input)
        # 谓词embdedding
        pred_input = Input(shape=(self.max_words,), dtype='int32', name='pred_input')
        pred_emb = Embedding(self.word_vocab_size, self.word_emb_dim, input_length=self.max_words, name='pred_emb',
                             mask_zero=True)(pred_input)
        # 融合拼接成一个输出
        total_input = keras.layers.concatenate([word_emb, pre_word_input, pos_emb, pred_emb], name='total_input')
        # WORD_DROPOUT
        emb_droput = Dropout(dropout)(total_input)
        # k层BILSTM
        bilstm_out = self.add_BILSTM(emb_droput)
        # DROPOUT
        # BILSTM输出的Dropout层
        bilstm_dropout = Dropout(dropout)(bilstm_out)
        # 全连接总结
        dense = TimeDistributed(Dense(self.role_nums), name='Dense')(bilstm_dropout)
        # softmax概率归一化
        softmax = TimeDistributed(Activation('softmax'), name='Softmax')(dense)
        # CRF分类
        crf = CRF(self.role_nums, name='CRF', sparse_target=True)
        crf_output = crf(softmax)

        # 汇总
        self.model = Model(inputs=[word_input, pre_word_input, pos_input, pred_input], outputs=[crf_output])

    def summary(self,printout = False, fn=False, pic=False):
        """
         网络信息汇总，三个控制分别是：
         printout：是否输出到控制台
         fn：是否输出到文件
         pic：是否把模型图输出
        """
        if printout:
            self.model.summary()  # 打印汇总信息
        if fn:
            with open('model_info/struct/'+self.model_name+'.txt', 'w') as fh:
                # Pass the file handle in as a lambda function to make it callable
                self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
        if pic:
            plot_model(self.model, to_file='model_info/struct/'+self.model_name+'.png', show_shapes=False,
                       show_layer_names=True)  # 绘制结构图

    def save_model_struct(self):
        # 保存模型结构
        model_json = self.model.to_json()
        with open("model_info/struct/"+self.model_name+".json", "w") as json_file:
            json_file.write(model_json)

    def load_model_struct(self):
        # 载入模型结构
        with open("model_info/struct/"+self.model_name+".json", "r") as json_file:
            model_json = json_file.readline()
        self.model = model_from_json(model_json, custom_objects={'CRF':CRF})

    def load_model_weights(self, fn):   # 权重文件参数
        # 载入模型权重
        self.model.load_weights("model_info/weights/%s-%s.hd" % (self.model_name, fn))

    # 用于添加多层LSTM
    def add_BILSTM(self, input):
        now_input = input
        for i in range(self.bilstm_num):
            # 第一层BILSTM
            now_input = Bidirectional(LSTM(self.lstm_dim, return_sequences=True))(now_input)
        return now_input

    # 训练模型
    def train(self, x, y, batch_size=32, epochs=50, validsplit = 0.1):
        crf = self.model.layers[-1]
        # 译模型， 学习率默认0.001, loss函数：负对数似然
        self.model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])
        # 训练模型
        early_stop = EarlyStopping('loss', verbose=1)  # 损失函数不再变好时终止
        # 保留每个epoch结束后的权重
        checkepoint = ModelCheckpoint("model_info/weights/%s-{epoch:02d}.hd" % self.model_name, monitor='loss',
                                      save_weights_only=True)
        # 把每轮训练结果记录到CSV文件里
        csvlogger = CSVLogger("output/logs/%s-logs.csv" % self.model_name)
        # 随情况调整学习率
        changelr = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, epsilon=0.01)
        # 默认会随机打乱样本，所以是交叉验证
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1,
                  callbacks=[early_stop, checkepoint, changelr, csvlogger]
                  , validation_split=validsplit)

    def output2file(self, filepath, inputfile, output, vocabs):
        # 把模型的输出转换成评测脚本可以识别的输出
        sents = open(inputfile, "r").readlines()
        tri_words = [sent.split() for sent in sents]  # 每个元素是A/B/C
        tri_words = [sent for sent in tri_words if len(sent) > 0]
        words = []
        for sent in tri_words:
            words.append([tri.split("/") for tri in sent])
        output = [to_index(sent) for sent in output]
        output = [vocabs['roles'].decode_sequence(sent) for sent in output]
        for i, sent in enumerate(words):
            for j, word in enumerate(sent):
                if (len(word) == 2):
                    word.append(output[i][j])
                elif (len(word) == 3):
                    word[2] = output[i][j]
                else:
                    raise Exception("Invalid word！")
        tri_words = []
        for sent in words:
            tri_words.append(["/".join(w) for w in sent])
        sents = [" ".join(sent) + "\n" for sent in tri_words]
        open(filepath, "w").writelines(sents)

    def test(self, x, filepath, typ, vocabs, batch_size=32):
        # 使用模型进行预测
        output = self.model.predict(x, batch_size=batch_size, verbose=1)
        np.save("output/" + typ + "-" + self.model_name + ".npy", output)
        # output = np.load("output/"+typ+"-"+MODEL_NAME+".npy")
        self.output2file("output/" + typ + "-" + self.model_name + ".txt", filepath, output, vocabs)

    def judge(self, input, typ, filepath, vocabs, batch_size=32):
        """
        评估模型
        """
        self.test(input, filepath, typ, vocabs, batch_size)
        print u'%s结果为。。。' % typ
        # check.check("output/" + typ + "-" + self.model_name + ".txt")
        try:
            res = J.calc_f1("output/" + typ + "-" + self.model_name + ".txt", filepath)
            print res
            open('model_info/struct//' + self.model_name + '.txt', "a").write(("%s结果为：\n" % typ) + res)
        except Exception, e:
            print e



