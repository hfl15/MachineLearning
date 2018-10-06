from base import *


def calEntropy(labels):
    n = float(len(labels))
    uniqueClass = np.unique(labels)
    entropy = 0.0
    for ilabel in uniqueClass.tolist():
        num = np.sum(labels == ilabel)
        prob = float(num) / n
        entropy += -prob * np.log2(prob)

    return entropy


def splitDataSet(splitClumn):
    uniqueVals = np.unique(splitClumn)
    subSetIndsDic = {}
    indexs = np.array(range(len(splitClumn)))
    for val in uniqueVals.tolist():
        subSetIndsDic[val] = indexs[splitClumn == val]

    return subSetIndsDic



def selectBestFeatureToSplit(dataArr, labels):
    m, n = dataArr.shape
    baseEntropy = calEntropy(labels)
    bestInfoGain = 0.0
    bestFeatureIndex = -1
    for i in range(n):
        splitColumn = dataArr[:, i]
        subSetIndsDic = splitDataSet(splitColumn)
        curEntropy = 0.0
        for key, inds in subSetIndsDic.items():
            prob = float(len(inds)) / float(m)
            curEntropy += prob * calEntropy(labels[inds])

        curInfoGain = baseEntropy - curEntropy
        if curInfoGain > bestInfoGain:
            bestInfoGain = curInfoGain
            bestFeatureIndex = i

    return bestFeatureIndex


def createTree(dataArr, labels, minCount, featureNames):

    # termination
    if len(np.unique(labels)) == 1:
        return labels[0]
    if len(labels) <= minCount:
        return majorityVote(labels)

    # grow tree
    bestFeatureIndex = selectBestFeatureToSplit(dataArr, labels)
    bestFeatureName = featureNames[bestFeatureIndex]

    subSetDic = splitDataSet(dataArr[:, bestFeatureIndex])
    treeRoot = {bestFeatureName: {}}
    featureNames = np.delete(featureNames, bestFeatureIndex)
    dataArr = np.delete(dataArr, bestFeatureIndex, 1)

    for key, inds in subSetDic.items():
        subDataArr = dataArr[inds]
        subLabels = labels[inds]
        treeRoot[bestFeatureName][key] = createTree(subDataArr, subLabels, minCount, featureNames)

    return treeRoot



def testDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    dataSet = np.array(dataSet)
    featureNameList = ['no surfacing', 'flippers']
    return dataSet[:, :-1], dataSet[:, -1], featureNameList


if __name__ == '__main__':
    dataArr, labels, featureNameList = testDataSet()
    treeDic = createTree(dataArr, labels, 1, featureNameList)
    print(treeDic)

