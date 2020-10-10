from pprint import pprint


def find_depth(tree, depth):
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


tree = {'x4': {0: {'x8': {0.0: {'x1': {0.0: 0.0,
                                1.0: {'x14': {0.0: 0.0,
                                              1.0: {'x12': {0.0: 0.0,
                                                            1.0: 1.0}}}}}},
                   1.0: {'x1': {0.0: {'x6': {0.0: 0.0,
                                             1.0: {'x5': {0.0: {'x3': {0.0: 0.0,
                                                                       1.0: 1.0}},
                                                          1.0: 1}}}},
                                1.0: {'x11': {0.0: {'x13': {0.0: 0.0,
                                                            1.0: {'x15': {0.0: {'x20': {0.0: {'x17': {0.0: 1.0,
                                                                                                      1.0: 0.0}},
                                                                                        1.0: 0}},
                                                                          1.0: {'x17': {0.0: 0.0,
                                                                                        1.0: 1.0}}}}}},
                                              1.0: {'x12': {0.0: {'x16': {0.0: 1.0,
                                                                          1.0: 0.0}},
                                                            1.0: 1}}}}}}}},
        1: {'x13': {0.0: {'x1': {0.0: {'x11': {0.0: 1.0,
                                               1.0: {'x7': {0.0: 0.0,
                                                            1.0: 1.0}}}},
                                 1.0: {'x12': {0.0: 0.0,
                                               1.0: {'x11': {0.0: 0.0,
                                                             1.0: 1.0}}}}}},
                    1.0: {'x11': {0.0: {'x7': {0.0: {'x2': {0.0: 0.0,
                                                            1.0: {'x15': {0.0: 0.0,
                                                                          1.0: 1.0}}}},
                                               1.0: {'x10': {0.0: {'x19': {0.0: {'x20': {0.0: 1.0,
                                                                                         1.0: 0.0}},
                                                                           1.0: 1}},
                                                             1.0: 0.0}}}},
                                  1.0: {'x9': {0.0: {'x18': {0.0: 1.0,
                                                             1.0: 0.0}},
                                               1.0: 1}}}}}}}}
pprint(tree)
print(find_depth(tree, 1) - 1)
