# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 19:27:02 2018

@author: techwiz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

df = joblib.load('df.pkl')
df_test = joblib.load('df_test.pkl')

X = df.iloc[:,:].values
target = joblib.load('target.pkl')
x_test = df_test.iloc[:,:].values

clf_rf = joblib.load('random_forest.sav')
clf_gbm = joblib.load('lightgbm.sav')
clf_xgboost = joblib.load('xgboost.sav')
clf_lr = joblib.load('logistic.sav')
clf_nb = joblib.load('GaussianNB.sav')
clf_dtc = joblib.load('decision_tree.sav')

y_pred1 = joblib.load('y_pred1.pkl')

# To be Done
from sklearn.ensemble import VotingClassifier

eclf = VotingClassifier( estimators = [ ('rf',clf_rf),('gbm',clf_gbm) ,('xg',clf_xgboost),('lr',clf_lr),('nb',clf_nb),('dtc',clf_dtc)],voting='soft',weights=[2,1.4,1.4,1,1,1])
eclf.fit(X,target)
epred = eclf.predict(x_test)
joblib.dump(eclf,'eclf2.sav')
joblib.dump(epred,'epred2.pkl')

sub = pd.read_csv("sample_submission.csv")
sub['damage_grade'] = epred
encode_ = { 5:'Grade 5', 4:'Grade 4',3:'Grade 3',2:'Grade 2',1:'Grade 1'}
sub['damage_grade'].replace(encode_,inplace=True)
sub.to_csv("submission_ensemble2.csv",index=False)