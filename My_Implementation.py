import Decision_trees
import pandas as pd
import numpy as np
from pprint import pprint
import random
from itertools import chain
import pydot
from flatten_dict import flatten


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


def runID3algorithmtreePruning(data, features, compute_attr="y", leaf=None, maxdepth=None, tree=None, currdepth=0):
    if maxdepth is not None:
        if tree is not None:
            if maxdepth == currdepth:
                truecnt = 0
                flscnt = 0
                for i, v in data[compute_attr].items():
                    if v == 1:
                        truecnt = truecnt + 1
                    elif v == 0:
                        flscnt = flscnt + 1
                if truecnt > flscnt:
                    return 1
                else:
                    return 0
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
        currdepth = currdepth + 1
        for value in np.unique(data[max_info_gain_attr]):
            sub_data = data.where(data[max_info_gain_attr] == value).dropna()
            subtree = runID3algorithmtreePruning(sub_data, features, compute_attr, leaf, maxdepth, tree, currdepth)
            tree[max_info_gain_attr][value] = subtree
        return tree


def runID3algorithmtreeSamplePruning(data, features, compute_attr="y", leaf=None, maxdepth=None, tree=None, currdepth=0):
    if maxdepth is not None:
        if tree is not None:
            if maxdepth == currdepth:
                truecnt = 0
                flscnt = 0
                for i, v in data[compute_attr].items():
                    if v == 1:
                        truecnt = truecnt + 1
                    elif v == 0:
                        flscnt = flscnt + 1
                if truecnt > flscnt:
                    return 1
                else:
                    return 0
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
        currdepth = currdepth + 1
        for value in np.unique(data[max_info_gain_attr]):
            sub_data = data.where(data[max_info_gain_attr] == value).dropna()
            subtree = runID3algorithmtreePruning(sub_data, features, compute_attr, leaf, maxdepth, tree, currdepth)
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


def test_any_tree(tree, data):
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


def Run(part):
    data = None
    if part == 'part1':
        data = Decision_trees.genTrainingData(feature_dimension, training_data_points)
    elif part == 'part2':
        data = Decision_trees.genPruningData(training_data_points)
    else:
        return None
    df = pd.DataFrame.from_dict(data)
    pprint(df)
    # train, test = splitData(df, spit_variable)
    # df = pd.read_csv('D:/Study/ML/Decision_trees/data.csv')
    tree = runID3algorithmtreePruning(df, df.columns[:-1], maxdepth=prune_depth)
    pprint(tree)
    print(find_depth(tree))
    yvalues = test_data_tree(df, tree)
    succescnt = 0
    for i, v in yvalues.iterrows():
        if v['yvalues'] == df.iloc[i]['y']:
            succescnt = succescnt + 1
    true_data = None
    true_error = 0
    if part == 'part1':
        true_data = Decision_trees.genTrainingData(feature_dimension, true_testing_data_points)
    elif part == 'part2':
        true_data = Decision_trees.genPruningData(true_testing_data_points)
    true_error = test_any_tree(tree, true_data)
    train_error = succescnt * 100 / len(df.index)
    print('True error is ' + str(true_error))
    print('Training error is ' + str(train_error))
    noise = str(findNoiseinTree(tree))
    print('Noise in tree is ' + noise)
    return train_error - true_error, noise


def test_multiple_datapoints():
    global training_data_points, feature_dimension, true_testing_data_points, prune_depth
    feature_dimension = 10
    true_testing_data_points = 1000
    store_dict = []
    training_data_points = 100
    prune_depth = None
    # 10000,100.0,2.5019857029388404
    # 20000,100.0,1.122754491017964
    for m in range(10, 2000, 100):
        training_data_points = m
        diff, noise = Run('part2')
        store_dict.append(str(m) + ',' + str(diff) + ',' + str(noise))
    for v in store_dict:
        print(v)


def findNoiseinTree(tree):
    flattened_list = (flatten(tree))
    true_list = []
    spurious_list = ['x16', 'x17', 'x18', 'x19', 'x20']
    for key in flattened_list.keys():
        keylist = list(key)
        list1 = [ele for ele in keylist if ele not in [0, 0.0, 1, 1.0]]
        true_list.extend(list1)
    count = 0
    for val in spurious_list:
        count = count + true_list.count(val)
    return count * 100 / len(true_list)


def find_depth(tree, depth=0):
    depth = depth + 1
    if not isinstance(tree, dict):
        return depth
    treeLeft = {}
    treeRight = {}
    for node in tree.keys():
        if 0 in tree[node]:
            treeLeft = tree[node][0]
        else:
            treeLeft = None
        if 1 in tree[node]:
            treeRight = tree[node][1]
        else:
            treeRight = None
    val = max(find_depth(treeLeft, depth), find_depth(treeRight, depth))
    return val


def prune_tree_by_depth():
    store = []
    for i in [-1]:
        prune_depth = i
        df_train = pd.read_csv('D:/Study/ML/Decision_trees/data_pruning_train.csv')
        df_test = pd.read_csv('D:/Study/ML/Decision_trees/data_pruning_test.csv')
        tree = runID3algorithmtreePruning(df_train, df_train.columns[:-1], maxdepth=prune_depth)
        pprint(tree)
        print(find_depth(tree))
        yvalues = test_data_tree(df_test, tree)
        succescnt = 0
        for i, v in yvalues.iterrows():
            if v['yvalues'] == df_test.iloc[i]['y']:
                succescnt = succescnt + 1
        test_accuracy = succescnt * 100 / len(df_test.index)
        print('Training error is ' + str(test_accuracy))
        store.append(str(prune_depth)+ ',' + str(100-test_accuracy))
    for i in store:
        print(i)

prune_tree_by_depth()
# test_multiple_datapoints()
# spit_variable = 80
# store = []
# for i in range(1, 100):
#     store.append(Run())
# number_list = np.array(store)
# (unique, counts) = np.unique(number_list, return_counts=True)
# frequencies = np.asarray((unique, counts)).T
# print(frequencies)
# header= []
# for i in range(1,21):
#     header.append(i)
# header.append('y')
# dftest = pd.read_csv('D:/Study/ML/Decision_trees/test.csv', sep=' ', header=None, names=header)
# print()
# tree = runID3algorithmtreePruning(dftest, dftest.columns[:-1], maxdepth=2)
# pprint(tree)
