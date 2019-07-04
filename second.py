# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 15:21:47 2018

@author: techwiz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
building_structure = pd.read_csv("Building_Structure.csv")

""" Exploratory Data Analysis And Data Preprocessing """
""" Merging building structure to train and test dataframe based on building id
"""
building_structure.head(15)
df = train.merge(building_structure,how="left",on=["building_id"])
df_test = test.merge(building_structure,how="left",on=["building_id"])

# decoding id variables to integer
id_ = []
for i in df['building_id'].values:
    s = str(i)
    int_ = int(s,16)
    id_.append(int_)
df['building_id'] = id_
id_ = []
for i in df_test['building_id'].values:
    s = str(i)
    int_ = int(s,16)
    id_.append(int_)
df_test['building_id'] = id_
df['building_id'].head()
df_test['building_id'].head()

# encoding categorial variables
df['damage_grade'].value_counts()
encode = { 'Grade 5':5, 'Grade 4':4,'Grade 3':3,'Grade 2':2,'Grade 1':1}
df['damage_grade'].replace(encode,inplace = True)
df['area_assesed'].value_counts()
encode_area ={ 'Not able to inspect' :0,'Interior':1,'Exterior':2,'Building removed':3,'Both':4}
df['area_assesed'].replace(encode_area,inplace=True)
df_test['area_assesed'].replace(encode_area,inplace=True)
target = df['damage_grade'].values
df.drop('damage_grade',axis=1,inplace=True)
#joblib.dump(target,'target.pkl')

#encoding
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
d = defaultdict(LabelEncoder)
t = defaultdict(LabelEncoder)
df = df.apply(lambda x: d[x.name].fit_transform(x))
df_test = df_test.apply(lambda x: t[x.name].fit_transform(x))

# Checking Feature Importances 
#plotting features importances
from sklearn.ensemble import RandomForestClassifier
"""
clf = RandomForestClassifier(bootstrap=True, class_weight=None,
            criterion='entropy', max_depth=150, max_features='sqrt',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=2,
            min_samples_split=8, min_weight_fraction_leaf=0.0,
            n_estimators=200, n_jobs=-1, oob_score=True,
            random_state=None, verbose=5, warm_start=False)
clf.fit(X,target)
plt.plot(clf1.feature_importances_)
plt.xticks(np.arange(X.shape[1]),df.columns.tolist(),rotation=90)
#Need to drop all has_geotechnical and has_superstructure features after the plot with less importance
"""
df.drop(['building_id','district_id_x','district_id_y'],axis=1,inplace=True)
df_test.drop(['building_id','district_id_x','district_id_y'],axis=1,inplace=True)
df.drop(['has_superstructure_other','has_superstructure_rc_engineered','has_superstructure_rc_non_engineered','has_superstructure_bamboo','has_superstructure_timber','has_superstructure_cement_mortar_brick','has_superstructure_mud_mortar_brick','has_superstructure_cement_mortar_stone','has_superstructure_stone_flag','has_superstructure_mud_mortar_stone','has_superstructure_adobe_mud'],axis=1,inplace=True)
df.drop(['has_geotechnical_risk','has_geotechnical_risk_fault_crack','has_geotechnical_risk_flood','has_geotechnical_risk_land_settlement','has_geotechnical_risk_landslide','has_geotechnical_risk_liquefaction','has_geotechnical_risk_other','has_geotechnical_risk_rock_fall'],axis=1 ,inplace=True)
df_test.drop(['has_superstructure_other','has_superstructure_rc_engineered','has_superstructure_rc_non_engineered','has_superstructure_bamboo','has_superstructure_timber','has_superstructure_cement_mortar_brick','has_superstructure_mud_mortar_brick','has_superstructure_cement_mortar_stone','has_superstructure_stone_flag','has_superstructure_mud_mortar_stone','has_superstructure_adobe_mud'],axis=1,inplace=True)
df_test.drop(['has_geotechnical_risk','has_geotechnical_risk_fault_crack','has_geotechnical_risk_flood','has_geotechnical_risk_land_settlement','has_geotechnical_risk_landslide','has_geotechnical_risk_liquefaction','has_geotechnical_risk_other','has_geotechnical_risk_rock_fall'],axis=1 ,inplace=True)



# Finding Optimal Parameters using GridSearchCV
# Ensembling Different Algorithms
joblib.dump(df,'df.pkl')
joblib.dump(df_test,'df_test.pkl')
df = joblib.load('df.pkl')
df_test = joblib.load('df_test.pkl')
target = joblib.load('target.pkl')

X = df.iloc[:,:].values
x_test = df_test.iloc[:,:].values
clf1 = RandomForestClassifier(bootstrap=True, class_weight=None,
            criterion='entropy', max_depth=150, max_features='sqrt',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=2,
            min_samples_split=8, min_weight_fraction_leaf=0.0,
            n_estimators=200, n_jobs=-1, oob_score=True,
            random_state=None, verbose=5, warm_start=False)
clf1.fit(X,target)
joblib.dump(clf1,'random_forest.sav')
clf = joblib.load('random_forest.sav')
y_pred = clf1.predict(x_test)
y_pred1 = clf.predict(x_test)
joblib.dump(y_pred1,'y_pred1.pkl')

import lightgbm as lgb
clf2 = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        learning_rate=0.05, max_depth=50, min_child_samples=20,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=200,
        n_jobs=-1, num_leaves=300, objective=None, random_state=None,
        reg_alpha=0.0, reg_lambda=0.0, silent=False, subsample=1.0,
        subsample_for_bin=200000, verbose=5,subsample_freq=0)
clf2.fit(X,target)
y_pred2 = clf2.predict(x_test)
joblib.dump(clf2,'lightgbm.sav')
joblib.dump(y_pred2,'y_pred2.pkl')

import xgboost as xg
clf3 = xg.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.5,
       colsample_bytree=1, gamma=0.1, learning_rate=0.09, max_delta_step=4,
       max_depth=4, min_child_weight=0.5, missing=None, n_estimators=200,
       n_jobs=-1, nthread=None, objective='binary:logistic',
       random_state=42, reg_alpha=0.1, reg_lambda=0.7,
       scale_pos_weight=0.5, seed=None, silent=True, subsample=0.9)
clf3.fit(X,target)
y_pred3 = clf3.predict(x_test)
joblib.dump(clf3,'xgboost.sav')
joblib.dump(y_pred3,'y_pred3.pkl')

from sklearn.linear_model import LogisticRegression
clf4 = LogisticRegression(warm_start=True,verbose=5)
clf4.fit(X,target)
y_pred4 = clf4.predict(x_test)
joblib.dump(clf4,'logistic.sav')
joblib.dump(y_pred4,'y_pred4.pkl')
"""
from sklearn.neighbors import KNeighborsClassifier
clf5 = KNeighborsClassifier(n_neighbors= 512)
clf5.fit(X,target)
y_pred5 = clf5.predict(x_test)
joblib.dump(clf5,'knn1.sav')
from sklearn.neighbors import KNeighborsClassifier
clf6 = KNeighborsClassifier(n_neighbors= 1024)
clf6.fit(X,target)
y_pred6 = clf6.predict(x_test)
joblib.dump(clf6,'knn2.sav')
joblib.dump(y_pred5,'y_pred5.pkl')
joblib.dump(y_pred6,'y_pred6.pkl')
"""
from sklearn.naive_bayes import GaussianNB
clf7 = GaussianNB()
clf7.fit(X,target)
y_pred7 = clf7.predict(x_test)
joblib.dump(clf7,'GaussianNB.sav')
joblib.dump(y_pred7,'y_pred7.pkl')

from sklearn.tree import DecisionTreeClassifier
clf8 = DecisionTreeClassifier(max_depth=8,min_samples_leaf=4,max_features = "auto" ,random_state=42 )
clf8.fit(X,target)
y_pred8= clf8.predict(x_test)
joblib.dump(clf8,'decision_tree.sav')
joblib.dump(y_pred8,'y_pred8.pkl')
from sklearn.svm import SVC
clf9 = SVC()
clf9.fit(X,target)
y_pred9 = clf9.predict(x_test)
joblib.dump(clf9,'svm.sav')
joblib.dump(y_pred9,'y_pred9.pkl')

# submission
sub = pd.read_csv("sample_submission.csv")
sub['damage_grade'] = y_pred
encode_ = { 5:'Grade 5', 4:'Grade 4',3:'Grade 3',2:'Grade 2',1:'Grade 1'}
sub['damage_grade'].replace(encode_,inplace=True)
sub.to_csv("submission_2.csv",index=False)