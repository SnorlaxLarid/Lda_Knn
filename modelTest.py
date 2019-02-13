import  KNN
import jieba
import Lda
import pandas as pd
import gensim
import numpy
import math
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

model = KNN.Model()
model.initModel()
nbrs = NearestNeighbors(n_neighbors=17, algorithm='ball_tree')
nbrs.fit(model.WordFea_pca)  # 构造BallTree，可以快速找出6个最近邻居，原理待学习

# 用PCA降维
pca = PCA(n_components=600)
testData_pca = pca.fit_transform(model.testData)
trainData_pca = pca.fit_transform(model.trainData)
distance_test,indices_test = nbrs.kneighbors(testData_pca)
distance_train,indices_train = nbrs.kneighbors(trainData_pca)

print("测试集中 分类为体育  分类为财经")
result_spt = 0
result_fin = 0
for i in range(0,498):
    spt_num = 0
    fin_num = 0
    for result in indices_test[i]:
        if(result <= 999):
            spt_num += 1
        else:
            fin_num += 1
    if(spt_num > fin_num):
        result_spt += 1
    else:
       result_fin += 1

print(result_spt,result_fin)

result_spt = 0
result_fin = 0
for i in range(498,996):
    spt_num = 0
    fin_num = 0
    for result in indices_test[i]:
        if(result <= 999):
            spt_num += 1
        else:
            fin_num += 1
    if(spt_num > fin_num):
        result_spt += 1
    else:
       result_fin += 1

print(result_spt,result_fin)


print("训练集中 分类为体育  分类为财经")
result_spt = 0
result_fin = 0
for i in range(0,1000):
    spt_num = 0
    fin_num = 0
    for result in indices_train[i]:
        if(result <= 999):
            spt_num += 1
        else:
            fin_num += 1
    if(spt_num > fin_num):
        result_spt += 1
    else:
       result_fin += 1

print(result_spt,result_fin)

result_spt = 0
result_fin = 0
for i in range(1000,2000):
    spt_num = 0
    fin_num = 0
    for result in indices_train[i]:
        if(result <= 999):
            spt_num += 1
        else:
            fin_num += 1
    if(spt_num > fin_num):
        result_spt += 1
    else:
       result_fin += 1

print(result_spt,result_fin)