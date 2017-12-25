# 文件结构
目录下有如下文件：
-- data
---- vocab
--------word_vectors_pre.txt    预定义的词向量
---- cpbdev.txt    开发集输入与答案
---- cpbtest.txt    测试集输入
---- cpbtrain.txt    训练集输入与答案
-- model_info
---- struct
-------- Final.png    模型的结构图
-------- Final.txt    模型的每层的信息
---- weights
-------- Final-39.hd    用于给出测试集答案的权重
-- calc_f1.py
-- model.py    定义了我们的BILSTM+CRF模型
-- preprocess.py    预处理代码，先预处理输入与答案，使其能为模型训练所用
-- test.py    测试模型代码
-- train.py    训练模型代码
-- utils.py    一些通用自定义函数
-- Vocab.py    生成词典的代码
-- test.sh    执行测试的脚本
-- train.sh    运行训练的脚本
-- README.MD    本文件，介绍代码概况和使用说明

# 使用说明

# 训练
进入当前目录后，在命令行里输入:

    bash train.sh

# 测试和评估
进入当前目录后，在命令行里输入:

    bash test.sh [--w <weightepoch> --t testtype]

无--w参数时，将先训练，再使用训练好的模型预测
有--w及其后所跟字符串两个参数时，使用`<weightepoch>`所指定的权重进行预测
> 注意：权重文件命名为：`模型名字-训练轮数.hd`，`<weightepoch>`只需要输入轮数的部分即可

当有--t参数时，将按指定type进行预测和评估，type共3种：
test: 对测试集预测
train：对训练集预测结果，并评估F值
dev：对开发集预测结果，并评估F值

若无--t参数，则默认其为test