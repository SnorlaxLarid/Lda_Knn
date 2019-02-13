import jieba
import Lda
import pandas as pd
import gensim
import numpy
import math
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

class Model:

    #降维前的训练数据集  格式 ： word1  word2 ....
    #                             num1  num2
    trainData = []
    #降维后的特征矩阵
    WordFea_pca = []
    #所有的特征词集
    vocalSet_module = []
    #降维前的测试数据集
    testData = []
    def getFeature(self,strList):
        ###获取对应语料的特征词表与概率，返回vocalSet, proList
        ### strList：未作处理的lda各主题的词汇分布

        wordsProSet = []
        for str in strList:
            word = str.split("+")
            wordsProSet.append(word)

        vocalSet = set([])
        vocalProList = []

        # 对LDA结果集进行裁剪，得到各词与概率
        for wordPro in wordsProSet:
            for word in wordPro:
                vocalProList.append([word.split('*')[1].split('"')[1], word.split("*")[0]])

        # 获取特征词空间
        for vocalPro in vocalProList:
            vocalSet = vocalSet | set([vocalPro[0]])

        # set转换位list保证顺序
        vocalSet = list(vocalSet)

        # 得到特征词对应的概率
        proList = [0] * len(vocalSet)

        for i in range(0, len(vocalSet)):
            for vocalPro in vocalProList:
                if (vocalSet[i] == vocalPro[0]):
                    proList[i] += float(vocalPro[1])

        return vocalSet, proList


    def initModel(self):
        ###初始化KNN分词
        ### 获得降维前的测试，训练集，降维后的训练集


        #停用词文件路径
        dir = "C://Users//larid//Desktop//Study//NLP//语料//"
        stop_words = "".join([dir, 'car.txt'])

        # 定义停用词
        stopwords = pd.read_csv(stop_words, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
        stopwords = stopwords['stopword'].values

        # 储存要进行处理的各个主题的词汇概率形如'0.026*"比赛" + 0.010*"球队" + 0.009*"中....'
        strList_finance = []

        lda_finance = gensim.models.ldamodel.LdaModel.load("finance.model")
        for topic in lda_finance.print_topics(num_topics=20, num_words=200):
            # print(topic[1])
            strList_finance.append(topic[1])

        strList_sport = []
        lda_sport = gensim.models.ldamodel.LdaModel.load("sport.model")
        for topic in lda_sport.print_topics(num_topics=20, num_words=200):
            # print(topic[1])
            strList_sport.append(topic[1])

        #分别获取两个类别的词汇及概率
        vocalSet_finance, proList_finance = self.getFeature(strList_finance)
        vocalSet_sport, proList_sport = self.getFeature(strList_sport)

        # 组成统一特征空间
        vocalSet_all = set([])

        for word in vocalSet_finance:
            vocalSet_all = vocalSet_all | set([word])

        for word in vocalSet_sport:
            vocalSet_all = vocalSet_all | set([word])

        vocalSet_all = list(vocalSet_all)
        self.vocalSet_module = vocalSet_all

        #计算某个词在两中文本中的概率的欧氏距离 （P1 - P2）*（P1 - P2），舍弃掉比较小的值
        vocal_distanse = [0.0] * len(vocalSet_all)
        #储存要舍弃的词语
        removeVocal = []
        #要舍弃的词语的个数（距离为0，因为初始化时就已经为0.0了）
        zeroCount = 0

        for i in range(0, len(vocalSet_all)):
            word = vocalSet_all[i]
            word_pro_fin = 0.0
            word_pro_spt = 0.0
            if (word in vocalSet_finance):
                index = vocalSet_finance.index(word)
                word_pro_fin = proList_finance[index]
            if (word in vocalSet_sport):
                index = vocalSet_sport.index(word)
                word_pro_spt = proList_sport[index]
            pro = numpy.square(word_pro_spt - word_pro_fin)
           # 保留距离大于 0.001*0.001的词
            if (pro != 0.0 and math.log10(pro) > -6.0):
                vocal_distanse[i] = math.log10(pro)
            else:
                removeVocal.append(vocalSet_all[i])
                zeroCount += 1

        for word in removeVocal:
            vocalSet_all.remove(word)
        for i in range(0, zeroCount):
            vocal_distanse.remove(0.0)


       #统计两类文本中各特征词的个数，得到特征矩阵
        number = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', '%']
        txtDir = "C://Users//larid//Desktop//Study//NLP//语料//八大类语料，各1500篇//体育//"
        start = 0
        end = 1000

        sentences = []
        Word_Fea = []
        for i in range(start, end):
            txt_fea = [0] * len(vocalSet_all)
            file_desc = "".join([txtDir, str(i) + ".txt"])
            with open(file_desc, 'r', encoding='utf-8') as filein:
                for line in filein:
                    try:
                        segs = jieba.lcut(line)
                        segs = [v for v in segs if not str(v).isdigit()]  # 去数字
                        for char in number:
                            segs = [v for v in segs if char not in str(v)]  # 去掉带百分号，小数点的数字
                        segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
                        segs = list(filter(lambda x: x not in stopwords, segs))  # 去掉停用词
                        sentences.append(segs)
                    except Exception:
                        print(line)
                        continue
            filein.close()
            # print(sentences)
            for segs in sentences:
                for word in segs:
                    if (word in vocalSet_all):
                        index = vocalSet_all.index(word)
                        # print(index)
                        txt_fea[index] += 1
            Word_Fea.append(txt_fea)
            sentences.clear()

        txtDir = "C://Users//larid//Desktop//Study//NLP//语料//八大类语料，各1500篇//财经//"
        start = 798977
        end = 799977

        sentences = []

        for i in range(start, end):
            txt_fea = [0] * len(vocalSet_all)
            file_desc = "".join([txtDir, str(i) + ".txt"])
            with open(file_desc, 'r', encoding='utf-8') as filein:
                for line in filein:
                    try:
                        segs = jieba.lcut(line)
                        segs = [v for v in segs if not str(v).isdigit()]  # 去数字
                        for char in number:
                            segs = [v for v in segs if char not in str(v)]  # 去掉带百分号，小数点的数字
                        segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
                        segs = list(filter(lambda x: x not in stopwords, segs))  # 去掉停用词
                        sentences.append(segs)
                    except Exception:
                        print(line)
                        continue
            filein.close()
            # print(sentences)
            for segs in sentences:
                for word in segs:
                    if (word in vocalSet_all):
                        index = vocalSet_all.index(word)
                        # print(index)
                        txt_fea[index] += 1
            Word_Fea.append(txt_fea)
            sentences.clear()


        self.trainData = Word_Fea
        #用PCA降维
        pca = PCA(n_components=600)
        self.WordFea_pca = pca.fit_transform(Word_Fea)

        # 降维后的各维度所反映的原始数据方差的比例和
        k = 0
        for i in range(0, len(pca.explained_variance_ratio_)):
            k += pca.explained_variance_ratio_[i]
        self.initTestData()
        return k


    def initTestData(self):
        ### 获得降维前的测试集

        # 停用词文件路径
        dir = "C://Users//larid//Desktop//Study//NLP//语料//"
        stop_words = "".join([dir, 'car.txt'])

        # 定义停用词
        stopwords = pd.read_csv(stop_words, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
        stopwords = stopwords['stopword'].values

        number = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', '%']
        txtDir = "C://Users//larid//Desktop//Study//NLP//语料//八大类语料，各1500篇//体育//"
        start = 1001
        end = 1499

        sentences = []
        Word_Fea_test = []
        for i in range(start, end):
            txt_fea = [0] * len(self.vocalSet_module)
            file_desc = "".join([txtDir, str(i) + ".txt"])
            with open(file_desc, 'r', encoding='utf-8') as filein:
                for line in filein:
                    try:
                        segs = jieba.lcut(line)
                        segs = [v for v in segs if not str(v).isdigit()]  # 去数字
                        for char in number:
                            segs = [v for v in segs if char not in str(v)]  # 去掉带百分号，小数点的数字
                        segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
                        segs = list(filter(lambda x: x not in stopwords, segs))  # 去掉停用词
                        sentences.append(segs)
                    except Exception:
                        print(line)
                        continue
            filein.close()
            # print(sentences)
            for segs in sentences:
                for word in segs:
                    if (word in self.vocalSet_module):
                        index = self.vocalSet_module.index(word)
                        # print(index)
                        txt_fea[index] += 1
            Word_Fea_test.append(txt_fea)
            sentences.clear()

        txtDir = "C://Users//larid//Desktop//Study//NLP//语料//八大类语料，各1500篇//财经//"
        start = 799978
        end = 800476

        sentences = []

        for i in range(start, end):
            txt_fea = [0] * len(self.vocalSet_module)
            file_desc = "".join([txtDir, str(i) + ".txt"])
            with open(file_desc, 'r', encoding='utf-8') as filein:
                for line in filein:
                    try:
                        segs = jieba.lcut(line)
                        segs = [v for v in segs if not str(v).isdigit()]  # 去数字
                        for char in number:
                            segs = [v for v in segs if char not in str(v)]  # 去掉带百分号，小数点的数字
                        segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
                        segs = list(filter(lambda x: x not in stopwords, segs))  # 去掉停用词
                        sentences.append(segs)
                    except Exception:
                        print(line)
                        continue
            filein.close()
            # print(sentences)
            for segs in sentences:
                for word in segs:
                    if (word in self.vocalSet_module):
                        index = self.vocalSet_module.index(word)
                        # print(index)
                        txt_fea[index] += 1
            Word_Fea_test.append(txt_fea)
            sentences.clear()

        self.testData = Word_Fea_test


