import os

import pandas as pd
from sklearn import model_selection



if __name__ == '__main__':
    files = pd.read_csv('mgmt_labels.csv', dtype=str)
    #检验数据是否存在
    id = files['BraTS21ID']
    root = 'F:/BraTS2021_Training_Data/'

    for name in id:
        path = root + 'BraTS2021_' + str(name)
        if not os.path.isdir(path):
            files.drop(files[files['BraTS21ID'] == name].index, inplace=True)


    # 将数据集拆分为训练集和测试集
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(files['BraTS21ID'], files['MGMT_value'], test_size=0.2, random_state=2024)
    train = files.sample(n=None, frac=0.8, replace=False, weights=None, random_state=2024, axis=None)
    test = pd.merge(files, train, how='left', indicator=True).query("_merge=='left_only'").drop('_merge', 1)

    mgmt_1 = train[train['MGMT_value'] == 1]
    mgmt_0 = train[train['MGMT_value'] == 0]
    print(mgmt_0)
    print(mgmt_1)
    print(len(test))

    train.to_csv('mgmt_train.csv', index=False)
    test.to_csv('mgmt_test.csv', index=False)