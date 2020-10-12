from pprint import pprint

import pandas as pd
import random
import numpy as np
import csv


def genTrainingData(k, m):
    dataset = {}
    wlist = {}
    den = 0
    for x in range(2, k + 1):
        den = den + pow(0.9, x)
    dataset['x' + str(1)] = []
    for x in range(2, k + 1):
        wlist[x] = (pow(0.9, x) / den)
        dataset['x' + str(x)] = []
    dataset['y'] = []
    # print(wlist)
    for i in range(1, m + 1):
        x1 = int(random.randint(0, 1))
        featureList = {1: x1}
        dataset['x' + str(1)].append(x1)
        for x in range(2, k + 1):
            check = int(np.random.binomial(1, 0.75, 1))
            if check:
                featureList[x] = (featureList[x - 1])
                dataset['x' + str(x)].append(featureList[x - 1])
            else:
                featureList[x] = (1 - featureList[x - 1])
                dataset['x' + str(x)].append(1 - featureList[x - 1])
        sum = 0
        y = None
        for x in range(2, k + 1):
            sum = sum + wlist[x] * featureList[x]
        if sum >= 1 / 2:
            y = featureList[1]
        else:
            y = 1 - featureList[1]
        featureList['y'] = y
        dataset['y'].append(y)
        # dataset.append(featureList)
    return dataset


def genPruningData(m):
    dataset = {}
    dataset['x' + str(1)] = []
    for x in range(2, 21):
        dataset['x' + str(x)] = []
    dataset['y'] = []
    for i in range(1, m + 1):
        x1 = int(random.randint(0, 1))
        featureList = {1: x1}
        dataset['x' + str(1)].append(x1)
        for x in range(2, 16):
            check = int(np.random.binomial(1, 0.75, 1))
            if check:
                featureList[x] = (featureList[x - 1])
                dataset['x' + str(x)].append(featureList[x - 1])
            else:
                featureList[x] = (1 - featureList[x - 1])
                dataset['x' + str(x)].append(1 - featureList[x - 1])
        for x in range(16, 21):
            dataset['x' + str(x)].append(int(random.randint(0, 1)))
        y = None
        if featureList[1] == 0:
            (vals, counts) = np.unique(list(featureList.values())[1:8], return_counts=True)
            ind = np.argmax(counts)
            y = vals[ind]
        else:
            (vals, counts) = np.unique(list(featureList.values())[8:15], return_counts=True)
            ind = np.argmax(counts)
            y = vals[ind]
        dataset['y'].append(y)
    return dataset



# data = genPruningData(10000)
# # pprint(data)
#
# data = genTrainingData(4, 10)
# pprint(data)
# data = pd.DataFrame.from_dict(data)
# pprint(data)
# data.to_csv('test.csv')
# # # print(data)
# trainfile = 'D:/Study/ML/Decision_trees/data_pruning_train.csv'
# testfile = 'D:/Study/ML/Decision_trees/data_pruning_test.csv'
# data = pd.DataFrame.from_dict(data)
# # # # pd.
# df_train = pd.DataFrame.from_dict(data.iloc[:8000].reset_index(drop=True))
# df_test = pd.DataFrame.from_dict(data.iloc[8000:].reset_index(drop=True))
# df_train.to_csv(trainfile, index=False)
# df_test.to_csv(testfile, index=False)
# with open(filename, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(data[0].keys())
#     for x in data:
#         csvwriter.writerow(x.values())
