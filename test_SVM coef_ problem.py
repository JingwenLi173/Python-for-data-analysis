import numpy as np
import sklearn

X = np.array([[3, 4], [1, 4], [2, 3], [6, -1], [7, -1], [5, -3]])
y = np.array([-1, -1, -1, 1, 1, 1])
from sklearn.svm import SVC

clf = SVC(C=1e5, kernel='linear')
clf.fit(X, y)
print('w = ', clf.coef_)
print('b = ', clf.intercept_)
print('Indices of support vectors = ', clf.support_)
print('Support vectors = ', clf.support_vectors_)
print('Number of support vectors for each class = ', clf.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))

# https://stats.stackexchange.com/questions/39243/how-does-one-interpret-svm-feature-weights

# Thus, the coef_ is just the weights "w" of the hyperplane "wTx+b=0" when distance between support vector and hyperplane equals 1. 
# The reason we use weights here as importance value is very similar to linear regression.  
# But this is only for linear kernel.
