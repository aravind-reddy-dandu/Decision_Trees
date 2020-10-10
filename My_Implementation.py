import Decision_trees
import pandas as pd
import numpy as np
from pprint import pprint
import random
from itertools import chain
import pydot


# Function to find H value for a given column. We need maximum of this
# Takes a pandas column as input
def getInformationContent(col):
    truecnt = 0
    flscnt = 0
    total = 0
    for i, v in col.items():
        if v == 1:
            truecnt = truecnt + 1
        elif v == 0:
            flscnt = flscnt + 1
        total = total + 1
    positiveVal = 0
    negVal = 0
    if truecnt != 0:
        positiveVal = (truecnt / total) * np.log2(truecnt / total)
    if flscnt != 0:
        negVal = (flscnt / total) * np.log2(flscnt / total)
    value = -(positiveVal + negVal)
    return value


def getInfoGain(attr_split, data, target):
    targetInfo = getInformationContent(data[target])
    # Now find conditional infocontent
    uniqueVals, count = np.unique(data[attr_split], return_counts=True)
    conditional_infocontent = np.sum(
        [(count[i] / np.sum(count)) * getInformationContent(
            data.where(data[attr_split] == uniqueVals[i]).dropna()[target])
         for i in range(len(uniqueVals))])
    infoGain = targetInfo - conditional_infocontent
    return infoGain


def runID3algorithmtree(data, features, compute_attr="y", leaf=None):
    if len(np.unique(data[compute_attr])) <= 1:
        return np.unique(data[compute_attr])[0]
    elif len(data) == 0:
        return random.randint(0, 1)
    elif len(features) == 0:
        return leaf
    else:
        truecnt = 0
        flscnt = 0
        for i, v in data[compute_attr].items():
            if v == 1:
                truecnt = truecnt + 1
            elif v == 0:
                flscnt = flscnt + 1
        if truecnt > flscnt:
            leaf = 1
        else:
            leaf = 0
        info_gains = {}
        for feature in features:
            info_gains[getInfoGain(feature, data, compute_attr)] = feature
        max_info_gain_attr = info_gains[max(info_gains.keys())]
        tree = {max_info_gain_attr: {}}
        features = [i for i in features if i != max_info_gain_attr]
        for value in np.unique(data[max_info_gain_attr]):
            sub_data = data.where(data[max_info_gain_attr] == value).dropna()
            subtree = runID3algorithmtree(sub_data, features, compute_attr, leaf)
            tree[max_info_gain_attr][value] = subtree
        return tree


# def splitData(data, size):
#     value = int((size * len(data.index)) / 100)
#     training_data = data.iloc[:value].reset_index(drop=True)
#     testing_data = data.iloc[value:].reset_index(drop=True)
#     return training_data, testing_data


def test_one_datapoint(feature, tree):
    for key in list(feature.keys()):
        if key in list(tree.keys()):
            result = tree[key][feature[key]]
            if isinstance(result, dict):
                return test_one_datapoint(feature, result)
            else:
                return result


def test_data_tree(data, tree):
    inputfeatures = data.iloc[:, :-1].to_dict(orient="records")
    yvalues = pd.DataFrame(columns=["yvalues"])
    for i in range(len(data)):
        yvalues.loc[i, "yvalues"] = test_one_datapoint(inputfeatures[i], tree)
    return yvalues


def test_any_tree(tree):
    data = Decision_trees.genTrainingData(feature_dimension, true_testing_data_points)
    data = pd.DataFrame.from_dict(data)
    inputfeatures = data.iloc[:, :-1].to_dict(orient="records")
    yvalues = pd.DataFrame(columns=["yvalues"])
    for i in range(len(data)):
        yvalues.loc[i, "yvalues"] = test_one_datapoint(inputfeatures[i], tree)
    succescnt = 0
    for i, v in yvalues.iterrows():
        if v['yvalues'] == data.iloc[i]['y']:
            succescnt = succescnt + 1
    return succescnt * 100 / len(data.index)


def Run():
    data = Decision_trees.genTrainingData(feature_dimension, training_data_points)
    df = pd.DataFrame.from_dict(data)
    # pprint(df)
    # train, test = splitData(df, spit_variable)
    # df = pd.read_csv('D:/Study/ML/Decision_trees/data.csv')
    tree = runID3algorithmtree(df, df.columns[:-1])
    pprint(tree)
    yvalues = test_data_tree(df, tree)
    succescnt = 0
    for i, v in yvalues.iterrows():
        if v['yvalues'] == df.iloc[i]['y']:
            succescnt = succescnt + 1
    # print(succescnt * 100 / len(test.index))
    # return list(tree.keys())[0]
    true_error = test_any_tree(tree)
    train_error = succescnt * 100 / len(df.index)
    print('True error is ' + str(true_error))
    print('Training error is ' + str(train_error))
    return train_error - true_error


def test_multiple_datapoints():
    global training_data_points, feature_dimension, true_testing_data_points
    feature_dimension = 10
    true_testing_data_points = 10000
    store_dict = []
    for m in range(100, 2501, 100):
        training_data_points = m
        diff = Run()
        store_dict.append(str(m) + ',' + str(diff))
    print(store_dict)


test_multiple_datapoints()
# spit_variable = 80
# store = []
# for i in range(1, 100):
#     store.append(Run())
# number_list = np.array(store)
# (unique, counts) = np.unique(number_list, return_counts=True)
# frequencies = np.asarray((unique, counts)).T
# print(frequencies)
