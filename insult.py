import pylab as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split,StratifiedKFold,cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
import sklearn.preprocessing as pp
from sklearn.manifold import Isomap
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

working_directory =r"./"

train_data = pd.read_csv(working_directory + "./train.csv")

test_data = pd.read_csv(working_directory + "./test_with_solutions.csv")


y_train = np.array(train_data.Insult)
comments_train = np.array(train_data.Comment)
print(comments_train.shape)
print(y_train.shape)

cv = CountVectorizer()
cv.fit(comments_train)
print(cv.get_feature_names()[:15])

X_train = cv.transform(comments_train).tocsr()
print("X_train.shape: %s" % str(X_train.shape))

svm = SVC()
param_grid = {'C': 10. ** np.arange(-3, 4)}
clf = GridSearchCV(svm, param_grid=param_grid, cv=3, verbose=3)
clf.fit(X_train, y_train)

comments_test = np.array(test_data.Comment)
y_test = np.array(test_data.Insult)
X_test = cv.transform(comments_test)
print(clf.score(X_test, y_test))

