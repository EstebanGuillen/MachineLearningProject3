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


df = pd.read_csv('train.data', header=None)


imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
df_imputed = pd.DataFrame(imr.transform(df.values))


X = df_imputed.loc[:, 1:].values
y = df_imputed.loc[:, 0].values


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1)


#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(random_state=1))])
#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', SVC(kernel='linear', C=10.0, random_state=1))])
#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', DecisionTreeClassifier(criterion='entropy',max_depth=40, random_state=0))])
#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', RandomForestClassifier(criterion='entropy',n_estimators=1000,n_jobs=4, random_state=1))])
#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'))])
#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)  )])
pipe_lr = Pipeline([('scl', StandardScaler()),('clf', GaussianNB()  )])

#pipe_lr.fit(X_train, y_train)
#print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))


scores = cross_val_score(estimator=pipe_lr, X = X_train, y=y_train, cv=10, n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))