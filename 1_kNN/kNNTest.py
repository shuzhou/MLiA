#coding=utf-8

__author__ = 'joe'

# from numpy import *
import numpy
import operator

def createDataSet():
    group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables

'''
kNN分类的主函数
@:param
    inX:待分类的测试数据
    dataSet:测试数据
    label：测试数据对应的标签
    k:kNN算法中的k近邻点数
'''

def classify0(inX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
    diffMat = numpy.tile(inX, (dataSetSize, 1)) - dataSet # tile(inX, (dataSetSize, 1)功能是将inX重复(dataSize,1)[dataSzie行，1列]次
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)#axis=0是按列求和,axis=1 是按行求和
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() #argsort返回的是排序后的数据的索引
    classCount = {}
    for i in range(k):
        voteIlable = lables[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1

    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    group, lables =  createDataSet()
    print classify0((0,0.002), group, lables, 3)