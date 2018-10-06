import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


DATA_ROOT = "../data/datingDataSet/"


def loadData(filename):
    fr = open(filename)
    dataList = []
    labelList = []
    for line in fr.readlines():
        lineList = line.strip().split('\t')
        dline = [float(x) for x in lineList[:-1]]
        dataList.append(dline)
        labelList.append(lineList[-1])

    return np.array(dataList), np.array(labelList)


def normalizeRange(dataArr):
    minVals = dataArr.min(axis=0)
    maxVals = dataArr.max(axis=0)
    ranges = maxVals - minVals
    n = len(dataArr)
    normData = (dataArr - np.tile(minVals, (n, 1))) / np.tile(ranges, (n, 1))
    return normData


def majorityVote(labels):
    cdic = {}
    for i in labels:
        cdic[i] = cdic.get(i, 0) + 1
    sortedCount = sorted(cdic.items(), key=lambda p: p[1], reverse=True)

    return sortedCount[0][0]