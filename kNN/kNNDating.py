#coding=utf-8

__author__ = 'joe'

import numpy
import matplotlib
import matplotlib.pyplot as plt
import kNNTest

'''
读取文件，将文件中的数据转换成matrix
'''

def file2matrix(fileName):
    fr = open(fileName)
    lines = fr.readlines()
    dataMatrix = numpy.zeros((len(lines), 3))
    lableVector = []
    for i in range(len(lines)):
        line = lines[i].strip()
        lineList = line.split("\t")
        dataMatrix[i,:] = lineList[0:3]
        lableVector.append(int(lineList[-1]))

    return dataMatrix,lableVector

"""
可视化数据
"""
def plotData(dataMatrix, labelVector):
    figure = plt.figure()
    ax = figure.add_subplot(111)#参数若是xyz则代表整个图分为x行y列（共计x*y个作图区域）在第z个区域作图
    ax.scatter(dataMatrix[:,2], dataMatrix[:,0], s = 15.0 * numpy.array(labelVector) ,c = 15.0 * numpy.array(labelVector)) #s代表点的大小 c代表点的颜色
    plt.show()

'''
数据归一化
@:return
normDataMartix：归一化后的数据
ranges：数值范围
minVals:最小值
'''

def autoNorm(dataMatrix):
    minVals = dataMatrix.min(0)
    maxVals = dataMatrix.max(0)
    ranges = maxVals - minVals
    normDataMartix = numpy.zeros(numpy.shape(dataMatrix))
    m = dataMatrix.shape[0]
    normDataMartix = dataMatrix - numpy.tile(minVals, (m,1))
    normDataMartix = normDataMartix / numpy.tile(ranges, (m,1))

    return normDataMartix, ranges, minVals

if __name__ == "__main__":
    hoRatio = 0.1
    dataMatrix, labelVector = file2matrix("./data/dating/datingTestSet2.txt")
    normDataMartix, ranges, minVals = autoNorm(dataMatrix)
    m = normDataMartix.shape[0]
    numTestData = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestData):
        classifyResult = kNNTest.classify0(normDataMartix[i, :], normDataMartix[numTestData:m, :], labelVector[numTestData:m] ,3)
        print "the classify result is: %d, the real result is: %d"%(classifyResult, labelVector[i])
        if(classifyResult != labelVector[i]):
            errorCount += 1
    print "the total error rate is :",errorCount/float(numTestData)
    print errorCount
