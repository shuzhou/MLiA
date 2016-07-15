#encoding : utf-8
__author__ = 'joe'

import os
import numpy
import kNNTest

"""
将图片文件转转换成矩阵
"""

def img2Vetor(fileName):
    returnVector = numpy.zeros(1024)
    file = open(fileName)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            returnVector[i * 32 + j] = int(line[j])
    return returnVector

if __name__ == "__main__":
    #操作训练数据
    trainDataFileNames = os.listdir("./data/handWriting/trainingDigits")
    trainDataNum = len(trainDataFileNames)
    trainDataLableVector = []
    trainDataMatrix = numpy.zeros((trainDataNum, 1024))
    for i in range(trainDataNum):
        fileName = trainDataFileNames[i]
        lable = int(fileName.split("_")[0])
        trainDataLableVector.append(lable)
        trainDataMatrix[i,:] = img2Vetor("./data/handWriting/trainingDigits/" + fileName)

    # 操作测试数据
    testDataFileNames = os.listdir("./data/handWriting/testDigits")
    testDataNum = len(testDataFileNames)
    errorCount = 0
    for i in range(testDataNum):
        fileName = testDataFileNames[i]
        lable = int(fileName.split("_")[0])
        testDataVector = img2Vetor("./data/handWriting/testDigits/" + fileName)
        classifyResult = int(kNNTest.classify0(testDataVector, trainDataMatrix, trainDataLableVector,3))
        print "classify result is:", classifyResult, "real result is:", lable
        if(classifyResult != lable):

            errorCount += 1

    print "trainData nun:", trainDataNum
    print "testData num:", testDataNum
    print "errors:", errorCount
    print "error rate:", errorCount / float(testDataNum)

    pass