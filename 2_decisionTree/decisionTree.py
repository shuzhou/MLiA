# coding=utf-8

# import numpy
import operator
import math


def loadData(fileName):
    """
    加载文件
    :param fileName: 文件名
    :return:
        dataMatrix : 数据(含标签)
        attrLableVector: 每个属性的名称
    """
    file = open(fileName)
    lines = file.readlines()
    dataMatrix = [line.strip().split("\t") for line in lines]
    attrLableVector = ['age', 'prescript', 'astigmatic', 'tearRate']

    return dataMatrix, attrLableVector


def majorityCnt(lableList):
    """
    投票表决分类结果: 当决策树没有属性可供选择时,此时分类结果为多种,则需要进行投票表决分类结果
    :param lableList: 多个分类结果值
    :return: 投票的结果
    """
    classCount = {}
    for lable in lableList:
        if lable not in classCount.keys():
            classCount[lable] = 0 #不存在则新建项
        classCount[lable] += 1

    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def calcShannonEnt(dataMatrix):
    """
    计算数据集的香农熵
    :param dataMatrix: 待计算的数据集
    :return: 香农熵值
    """
    numEntries = len(dataMatrix)
    lableDic = {}
    for entry in dataMatrix:
        lable = entry[-1]
        if lable not in lableDic.keys():
            lableDic[lable] = 0
        lableDic[lable] += 1
    shannonEnt = 0.0
    for key in lableDic:
        prob = float(lableDic[key])/numEntries
        shannonEnt -= prob * math.log(prob, 2)

    return shannonEnt


def splitDataMatrix(dataMatrix, index, value):
    """
    按照给定的特征索引及特征值
    :param dataMatrix: 待划分的数据集
    :param index: 索引坐标
    :param value: 对应索引坐标的特征值
    :return:
    """
    returnDataMatrix = []
    for entry in dataMatrix:
        if entry[index] == value:
            reducedFeatureVec = entry[:index]
            reducedFeatureVec.extend(entry[index+1:])
            returnDataMatrix.append(reducedFeatureVec)
    return returnDataMatrix

def chooseBestFeatureToSplit(dataMatrix):
    """
    选择最好的(信息增益最大的属性)数据集划分方式,即计算出最好的划分数据集的特征
    :param dataMatrix: 待划分的数据集
    :return: 信息增益最大的属性的索引
    """
    numFeatures = len(dataMatrix[0]) - 1
    baseEntropy = calcShannonEnt(dataMatrix)
    bestInfoGain = 0.0
    bestFeatureIndex = -1
    for i in range(numFeatures):
        featureList = [entry[i] for entry in dataMatrix]
        featureSet = set(featureList)
        newEntropy = 0.0
        for value in featureSet:
            subDataSet = splitDataMatrix(dataMatrix, i, value)
            prob = len(subDataSet) / float(len(dataMatrix))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy

        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeatureIndex = i

    return bestFeatureIndex


def createDecisionTree(dataMatrix, attrLableVector):
    """
    递归构造决策树
    :param dataMatrix:
    :param attrLableVector:
    :return:
    """
    classLableList = [entry[-1] for entry in dataMatrix]
    if(classLableList.count(classLableList[0]) == len(classLableList)):#类别完全相同,则停止划分
        return  classLableList[0]
    if(len(dataMatrix[0]) == 1):#没有属性供继续构造决策树,则投票表决分类标签
        return majorityCnt(classLableList)

    bestFeatureIndex = chooseBestFeatureToSplit(dataMatrix)
    bestFeatureLable = attrLableVector[bestFeatureIndex]
    myTree = {bestFeatureLable:{}}
    del (attrLableVector[bestFeatureIndex])
    featureValueList = [entry[bestFeatureIndex] for entry in dataMatrix]
    featureValueSet = set(featureValueList)
    for value in featureValueSet:
        subLables = attrLableVector[:] #传引用 防止
        myTree[bestFeatureLable][value] = createDecisionTree(splitDataMatrix(dataMatrix,bestFeatureIndex, value), subLables)

    return myTree

def plotTree(tree):
    """
    t图形化展示树
    :param tree:
    :return:
    """
    pass

if __name__ == "__main__":
    dataMatrix, attrLableVector = loadData("./data/lenses.txt")
    # print dataMatrix, attrLableVector
    myTree = createDecisionTree(dataMatrix,attrLableVector)
    print myTree
    plotTree(myTree)
