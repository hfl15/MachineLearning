from base import *

#######################################################################################
# basic function
#######################################################################################

def euclidean(X, Y):
    assert X.shape[0] == Y.shape[0] and X.shape[1] == Y.shape[1], "two vector didn't align"
    Err = X - Y
    ErrSqureSum = np.sqrt(np.sum(Err**2, axis=1))

    return ErrSqureSum




#######################################################################################
# knn classification
#######################################################################################

def classify(inXVec, dataArr, labels, k, distFunc=euclidean, voteFunc=majorityVote):
    n, m = dataArr.shape

    X = np.tile(inXVec, (n, 1))
    distArr = distFunc(X, dataArr)

    sortedInds = distArr.argsort()
    topK = labels[sortedInds[:k]]

    return voteFunc(topK)


def classifyBatch(testArr, trainArr, testLabels, trainLabels, k=5, distFunc=euclidean, voteFunc=majorityVote):
    n = len(testArr)
    yHats = list(map(lambda xVec: classify(xVec, trainArr, trainLabels, k, distFunc, voteFunc), testArr))
    err = np.sum(yHats != testLabels)

    return float(err) / float(n), yHats


def classifyBathOneSet(dataArr, labels, testRatio = 0.1, k=5, distFunc=euclidean, voteFunc=majorityVote):
    m, n = dataArr.shape
    lenTest = np.int(m*testRatio)
    indexs = range(m)

    testArr = dataArr[indexs[:lenTest]]
    testLabels = labels[indexs[:lenTest]]

    trainArr = dataArr[indexs[lenTest::]]
    trainLabels = labels[indexs[lenTest::]]

    return classifyBatch(testArr, trainArr, testLabels, trainLabels, k, distFunc, voteFunc)[0]


#######################################################################################
# test
#######################################################################################

# dating data set

def testDatingDataSet():
    # prepare data
    filename = DATA_ROOT + "datingTestSet.txt"
    dataArr, labels = loadData(filename)
    dataArr = normalizeRange(dataArr)

    # shuffle
    m, n = dataArr.shape
    indexs = list(range(m))
    np.random.shuffle(indexs)
    dataArr = dataArr[indexs]
    labels = labels[indexs]

    # split data set
    testRatio = 0.1
    lenTest = np.int(m * testRatio)
    testArr = dataArr[indexs[:lenTest]]
    testLabels = labels[indexs[:lenTest]]
    trainArr = dataArr[indexs[lenTest::]]
    trainLabels = labels[indexs[lenTest::]]

    # test different k
    kList = [1, 5, 10, 15, 20, 30, 50, 100, 200, 300, 400, 500]
    errs = []
    for k in kList:
        errRate, _ =  classifyBatch(testArr, trainArr, testLabels, trainLabels, k)
        errs.append(errRate)

    print("k", kList)
    print("err", errs)
    plt.plot(kList, errs)
    plt.xlabel('k')
    plt.ylabel('error')
    plt.show()

    # test different test size
    testRatios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    trainSizeList = []
    errs = []
    k = 100
    for ratio in testRatios:
        lenTest = np.int(m * ratio)
        testArr = dataArr[indexs[:lenTest]]
        testLabels = labels[indexs[:lenTest]]
        trainArr = dataArr[indexs[lenTest::]]
        trainLabels = labels[indexs[lenTest::]]
        errRate, _ = classifyBatch(testArr, trainArr, testLabels, trainLabels, k)

        errs.append(errRate)
        trainSizeList.append(m - lenTest)

    print("k: ", k)
    print("trainSet size: ", trainSizeList)
    print("erros", errs)
    plt.plot(trainSizeList, errs)
    plt.xlabel("train set size")
    plt.ylabel("error")
    plt.show()



if __name__ == '__main__':
    print("knn classification for dating data set.....")
    testDatingDataSet()