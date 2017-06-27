from  __future__ import division

import numpy
import pandas
import operator
import os
from sklearn import linear_model
from sklearn.preprocessing import scale
from scipy import special

COL = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age']

SAHEART_PATH = os.getcwd() + '/SAheartdata/SAheart.csv'

SAHEART_TEST = pandas.read_csv(SAHEART_PATH)

# Feature matrix for full model
X = SAHEART_TEST.ix[:, :9]

# Class labels
y = SAHEART_TEST.ix[:, 9]

# Change famhist from string to 0/1
X['famhist'] = X['famhist'].transform(lambda x: 1 if x == 'Present' else 0)

# A dataframe of zeros for the null model
X_null = pandas.DataFrame(numpy.zeros(y.shape).astype(int), columns=['Null'])

""" Task A """

# Create a null model
null_model = linear_model.LogisticRegression(C=1e6, solver='liblinear', fit_intercept=True, intercept_scaling=1e3)
null_model = null_model.fit(X_null, y)


""" Task B """

# Create models for each feature, Fit models for each feature
# Store models in a dictionary
single_feature_models = {}
for feature in X.columns:
    single_feature_models[feature] = linear_model.LogisticRegression(C=1e6, solver='liblinear', fit_intercept=True, intercept_scaling=1e3)
    single_feature_models[feature] = single_feature_models[feature].fit(X[feature].values.reshape(-1,1), y.values)


""" Task C """

# P values for likelihood ratio test
def likelihood_ratio_test(full, reduced, df):
    """ Returns the p value for likelihood ratio test"""
    D = -2 * (reduced - full)
    return special.gammaincc(df/2, D/2)

def log_likelihood(model, X, y):
    """ Return the log likelihood"""
    LL = 0.0
    for (x, y_true) in zip(model.predict_log_proba(X), y):
        LL += x[y_true]
    return LL

# Calculate Null value Log likelihood
LL_null = log_likelihood(null_model, X_null, y)

# Create a dictionary for pvalues
pvalues = {}

# Calculate p values for all features
for feature in X.columns:
    LL_single = log_likelihood(single_feature_models[feature], X[feature].values.reshape(-1,1), y)
    pvalues[feature] = likelihood_ratio_test(LL_single, LL_null, 1)

# Sort pvalues
sorted_pvalues = sorted(pvalues.items(), key=operator.itemgetter(1))

print sorted_pvalues


""" Task D """

def model_compare(model_new,model_old, data_new, data_old, y):
    """ Compare models by calculating pvalue"""
    LL_old = log_likelihood(model_old, data_old, y)
    LL_new = log_likelihood(model_new, data_new, y)
    df = data_new.shape[1] - data_old.shape[1]
    return likelihood_ratio_test(LL_new, LL_old, df)

X_incremental = pandas.concat([X, X_null], axis=1)

# Add null model in the beginning with random pvalue to enforce selection
sorted_pvalues.insert(0, ['Null',0.04])

# Start with the null model
selected_features = ['Null']
incremental_model = linear_model.LogisticRegression(C=1e6, solver='liblinear', fit_intercept=True, intercept_scaling=1e3)
incremental_model = incremental_model.fit(X_incremental[selected_features], y)

# Test features with increasing pvalues to test imporvement in the model
old_score = incremental_model.score(X_incremental[selected_features], y)
for feature in sorted_pvalues[1:]:
    if( feature[1] < 0.05):
        selected_features.append(feature[0])
        new_model = linear_model.LogisticRegression(C=1e6, solver='liblinear', fit_intercept=True, intercept_scaling=1e3)
        new_model = new_model.fit(X_incremental[selected_features], y)
        #score = new_model.score(X_incremental[selected_features], y)
        #score = model_compare(new_model, incremental_model, X_incremental[selected_features], X_incremental[selected_features[:-1]], y)
        score = model_compare(new_model, null_model, X_incremental[selected_features], X_null, y)
        #print feature[0], "is tested and the pvalue is", score
        #if(old_score < score):
        if score<0.05:
            print feature[0], "is selected and the pvalue is", score
            incremental_model = new_model
            old_score = score
        else:
            selected_features.remove(feature[0])

print "\n\nLikelihood slected features :"
print 'Intercept:', incremental_model.intercept_[0]
for (feature,coefficient) in zip(selected_features[1:], incremental_model.coef_[0,1:]):
    print feature,':',coefficient

""" Task E """

# Perform preprocessing on X, to scale features
X_scaled = pandas.DataFrame(scale(X),columns=COL)

# Full model with L1 loss
full_model = linear_model.LogisticRegression(C=0.05, solver='liblinear', fit_intercept=True, intercept_scaling=1e3, penalty='l1')
full_model = full_model.fit(X_scaled, y)

full_model_coef = {}
for (col,coefficient) in zip(COL,full_model.coef_[0]):
    full_model_coef[col] = coefficient
sorted_coef = sorted(full_model_coef.items(), key=operator.itemgetter(1), reverse=True)


print "\n\nL1 slected features :"
print 'Intercept:', full_model.intercept_[0]
for coefficient in sorted_coef:
    if coefficient[1] != 0:
        print coefficient[0],':', coefficient[1]
