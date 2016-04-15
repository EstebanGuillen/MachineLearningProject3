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
from scipy import stats


pd.set_option('mode.use_inf_as_null', True)

genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]


def classify(X,y,pipe,classifier_features):
    
    print ('')
    print ('')
    print('***** '+ classifier_features + ' *****')
    print('')

    skf = StratifiedKFold(y_train, n_folds=10, random_state=1)

    scores = []
    confusion_matrix = np.zeros((10,10))
    for train_index, test_index in skf:
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        pipe.fit(X_train_fold, y_train_fold)
        y_predict = pipe.predict(X_test_fold)
        scores.append(pipe.score(X_test_fold, y_test_fold))
    
        for p,r in zip(y_predict, y_test_fold):
            confusion_matrix[p,r] = confusion_matrix[p,r] + 1
        
    
    mean, sigma = np.mean(scores), np.std(scores)
    
    conf_int = stats.norm.interval(0.95, loc=mean, scale=sigma / np.sqrt(len(scores)))
    print('Average CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.mean(scores)-conf_int[0]))
    
    print('CV 95 percent confidence interval:', conf_int)
    print(confusion_matrix)


    confusion_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    pylab.clf()
    pylab.matshow(confusion_normalized, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels(genres)
    pylab.title("Cross Validation Confusion Matrix for " + classifier_features)
    pylab.colorbar()
    pylab.grid(False)
    #pylab.xlabel('Predicted class')
    #pylab.ylabel('True class')
    pylab.grid(False)
    pylab.show()
    
    print ('')
    
def train(X,y,X_test,y_test,pipe,classifier_features):
    classify(X,y,pipe, classifier_features)
    pipe.fit(X, y)
    print('Test Accuracy %s: %.6f' % (classifier_features, pipe.score(X_test, y_test)) )
    
df = pd.read_csv('train-fft-mfcc.data', header=None)
df = df.dropna(axis=0)

X = df.iloc[:, 1:].values

y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=300,
                              random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
top_50 = indices[:50]

pipe = Pipeline([('clf', RandomForestClassifier(criterion='entropy',n_estimators=3000,n_jobs=-1, random_state=1))])

#First feature set is just the fft features, first 1000 features (0-999) listed in the training data
X_train_fft, X_test_fft= X_train[:,:1000], X_test[:,:1000]

train(X_train_fft,y_train,X_test_fft,y_test,pipe, 'Random Forest (FFT)')
#classify(X_train_fft,y_train,pipe, "Cross Validation Confusion Matrix for Random Forest (FFT)")
#pipe.fit(X_train_fft, y_train)
#print('Test Accuracy Random Forest (FFT Features): %.6f' % pipe.score(X_test_fft, y_test))

#Second feature set is just the 13 mfcc features,  features 1000-1012 listed in the training data
X_train_mfcc, X_test_mfcc= X_train[:,1000:1013], X_test[:,1000:1013]
train(X_train_mfcc,y_train,X_test_mfcc,y_test,pipe, 'Random Forest (MFCC)')
#classify(X_train_mfcc,y_train,pipe, "Cross Validation Confusion Matrix for Random Forest (MFCC)")
#pipe.fit(X_train_mfcc, y_train)
#print('Test Accuracy Random Forest (MFCC Features): %.6f' % pipe.score(X_test_mfcc, y_test))


#Third feature set are the strongest 50 features from the set of FFT and MFCC features
X_train_top50, X_test_top50 = X_train[:,top_50], X_test[:,top_50]
train(X_train_top50,y_train,X_test_top50,y_test,pipe, 'Random Forest (Top 50 FFT and MFCC)')
#classify(X_train_top50,y_train,pipe, "Cross Validation Confusion Matrix for Random Forest (Top 50 FFT and MFCC)")
#pipe.fit(X_train_top50, y_train)
#print('Test Accuracy Random Forest (Top 50): %.6f' % pipe.score(X_test_top50, y_test))


pipe = Pipeline([('scl', StandardScaler()),('clf',SVC(kernel='rbf', C=100.0, random_state=1))])

#First feature set is just the fft features, first 1000 features (0-999) listed in the training data
X_train_fft, X_test_fft= X_train[:,:999], X_test[:,:999]
train(X_train_fft,y_train,X_test_fft,y_test,pipe, 'SVM (FFT)')
#classify(X_train_fft,y_train,pipe, "Cross Validation Confusion Matrix for SVM (FFT)")
#pipe.fit(X_train_fft, y_train)
#print('Test Accuracy SVM (FFT Features): %.6f' % pipe.score(X_test_fft, y_test))

#Second feature set is just the 13 mfcc features,  features 1000-1012 listed in the training data
X_train_mfcc, X_test_mfcc= X_train[:,1000:1013], X_test[:,1000:1013]
train(X_train_mfcc,y_train,X_test_mfcc,y_test,pipe, 'SVM (MFCC)')
#classify(X_train_mfcc,y_train,pipe, "Cross Validation Confusion Matrix for SVM (MFCC)")
#pipe.fit(X_train_mfcc, y_train)
#print('Test Accuracy SVM (MFCC Features): %.6f' % pipe.score(X_test_mfcc, y_test))


#Third feature set are the strongest 50 features from the set of FFT and MFCC features
X_train_top50, X_test_top50 = X_train[:,top_50], X_test[:,top_50]
train(X_train_top50,y_train,X_test_top50,y_test,pipe, 'SVM (Top 50 FFT and MFCC)')
#classify(X_train_top50,y_train,pipe, "Cross Validation Confusion Matrix for SVM (Top50 FFT and MFCC)")
#pipe.fit(X_train_top50, y_train)
#print('Test Accuracy SVM (Top 50): %.3f' % pipe.score(X_test_top50, y_test))


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


