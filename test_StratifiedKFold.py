import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[5,9],[1,5],[3,9],[5,8],[1,1],[1,4], [5,9],[1,5],[3,9],[5,8],[1,1],[1,4]])
y = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0,0,0,0,0,0,1])

print('X:',X)
print('y:',y)

skf = StratifiedKFold(n_splits=4,random_state=2020, shuffle=True)
print(skf)

for train_index, test_index in skf.split(X, y):
    print('TRAIN:', train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    

'''
X: [[1 2]
 [3 4]
 [1 2]
 [3 4]
 [5 9]
 [1 5]
 [3 9]
 [5 8]
 [1 1]
 [1 4]
 [5 9]
 [1 5]
 [3 9]
 [5 8]
 [1 1]
 [1 4]]
y: [0 1 1 1 0 0 1 0 0 0 0 0 0 0 0 1]
StratifiedKFold(n_splits=4, random_state=2020, shuffle=True)
TRAIN: [ 1  2  3  5  6  7  8  9 10 11 12 13] TEST: [ 0  4 14 15]
TRAIN: [ 0  2  3  4  5  6  9 10 11 13 14 15] TEST: [ 1  7  8 12]
TRAIN: [ 0  1  3  4  5  6  7  8 10 12 14 15] TEST: [ 2  9 11 13]
TRAIN: [ 0  1  2  4  7  8  9 11 12 13 14 15] TEST: [ 3  5  6 10]
'''

'''
n_splits：默认为3，表示将数据划分为多少份，即k折交叉验证中的k；
shuffle：默认为False，表示是否需要打乱顺序，这个参数在很多的函数中都会涉及，如果设置为True，则会先打乱顺序再做划分，如果为False，会直接按照顺序做划分；
random_state：默认为None，表示随机数的种子，只有当shuffle设置为True的时候才会生效。
'''
