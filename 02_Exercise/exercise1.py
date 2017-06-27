from  __future__ import division
import numpy as np
import pandas as pd
import sklearn.linear_model
from scipy import special


COL = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age', 'chd']

SAHEART_PATH = '/home/anshul/Documents/Study/Semester_2/MLMI/Exercises/02_Exercise/SAheartdata/SAheart.csv'

SAHEART_TEST = pd.read_csv(SAHEART_PATH, names=COL, skiprows=1)
SAHEART_TEST_DATA = SAHEART_TEST.values

for row in SAHEART_TEST_DATA:
    if row[4] == 'Present':
        row[4] = 1
    else:
        row[4] = 0

#Slice CHD into numpy array
y = SAHEART_TEST_DATA[:,9].astype(int)

#Slice everything else
X = SAHEART_TEST_DATA[:,:9]

#print np.zeros(y.size).reshape(-1,1)

#print X[:,0]

model = sklearn.linear_model.LogisticRegression(C=1e6,solver='liblinear',fit_intercept=True,intercept_scaling=1e3)



#print model.predict(X[:,0].reshape(-1,1))

# P values for likelihood ratio test
def likelihood_ratio_test(full, reduced, df):
    """ Returns the p value for likelihood ratio test"""
    D = -2 * (reduced - full)
    return special.gammaincc(df/2, D/2)

LLF_null = 0
null_model = model.fit(np.zeros(y.shape).reshape(-1,1), y)
for (x, y_true) in zip(null_model.predict_log_proba(np.zeros(y.shape).reshape(-1,1)), y):
    LLF_null += x[y_true]

for f in range(0,9):
    LLF_single = 0.0
    model = model.fit(X[:, f].reshape(-1, 1), y)
    for (x, y_true) in zip(model.predict_log_proba(X[:,f].reshape(-1,1)), y):
        LLF_single += x[y_true]

    print likelihood_ratio_test(LLF_single,LLF_null, 1)

#print X

#print y



mdl = model.fit(X, y)

#print model.score(X.reshape(-1,1), y)


#print mdl.predict_proba(X)

#print model.get_params(deep=True)

#print model.predict_proba(X)