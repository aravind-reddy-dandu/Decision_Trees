import pandas as pd
import numpy as np
import random
import My_Implementation
import Decision_trees
import pprint


training_data_points = 1000
def Run():
    data = Decision_trees.genPruningData(training_data_points)
    df = pd.DataFrame.from_dict(data)
    pprint(df)
    # train, test = splitData(df, spit_variable)
    # df = pd.read_csv('D:/Study/ML/Decision_trees/data.csv')
    tree = My_Implementation.runID3algorithmtree(df, df.columns[:-1])
    pprint(tree)
    yvalues = My_Implementation.test_data_tree(df, tree)
    succescnt = 0
    for i, v in yvalues.iterrows():
        if v['yvalues'] == df.iloc[i]['y']:
            succescnt = succescnt + 1
    # print(succescnt * 100 / len(test.index))
    # return list(tree.keys())[0]
    true_data = Decision_trees.genTrainingData(true_testing_data_points)
    true_error = test_any_tree(tree, true_data)
    train_error = succescnt * 100 / len(df.index)
    print('True error is ' + str(true_error))
    print('Training error is ' + str(train_error))
    return train_error - true_error



