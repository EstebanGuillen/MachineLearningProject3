import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from matplotlib import pylab
from sklearn.cross_validation import StratifiedKFold

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing


pd.set_option('mode.use_inf_as_null', True)

genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]


def classify(X,y,pipe):

   
    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1)


    skf = StratifiedKFold(y_train, n_folds=10)

    scores = []
    confusion_matrix = np.zeros((10,10))
    for train_index, test_index in skf:
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        pipe.fit(X_train_fold, y_train_fold)
        y_predict = pipe.predict(X_test_fold)
        scores.append(pipe.score(X_test_fold, y_test_fold))
    
        for p,r in zip(y_predict, y_test_fold):
            confusion_matrix[p,r] = confusion_matrix[p,r] + 1
        
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
    print(confusion_matrix)


    '''
    scores = cross_val_score(estimator=pipe, X = X_train, y=y_train, cv=10, n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))


    pipe.fit(X_train, y_train)
    print('Test Accuracy: %.3f' % pipe.score(X_test, y_test))
    y_predict = pipe.predict(X_test)
    confusion = confusion_matrix(y_true=y_test, y_pred=y_predict)
    print(confusion)
    '''
    confusion_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    #print(confusion_normalized)

    pylab.clf()
    pylab.matshow(confusion_normalized, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels(genres)
    #pylab.title("Confusion Matrix")
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(False)
    pylab.show()
    
    
    
df = pd.read_csv('train-fft-mfcc.data', header=None)
df = df.dropna(axis=0)

X = df.iloc[:, 1:].values

y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1)

#pipe = Pipeline([('scl', StandardScaler()), ('pca',PCA(n_components=10)) , ('clf', LogisticRegression(random_state=1))])
#pipe = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression())])
#pipe = Pipeline([('scl', StandardScaler()),('clf',SVC(kernel='linear', C=40.0, random_state=1))])
#pipe = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy',max_depth=50))])
pipe = Pipeline([('clf', RandomForestClassifier(criterion='entropy',n_estimators=1000,n_jobs=-1))])
#pipe = Pipeline([('scl', StandardScaler()),('clf', KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'))])
#pipe = Pipeline([('scl', StandardScaler()),('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)  )])
#pipe = Pipeline([('scl', StandardScaler()),('clf', GaussianNB()  )])

#classify(X,y,pipe)

# Build a forest and compute the feature importances
#X = preprocessing.scale(X)
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


top_50 = indices[:50]

X_train = X_train[:,top_50]
print(X_train.shape)
classify(X_train,y_train,pipe)

X_test = X_test[:,top_50]

pipe.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe.score(X_test, y_test))

'''
# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
'''


