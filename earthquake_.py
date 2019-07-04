# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 22:14:21 2018

@author: techwiz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
building_structure = pd.read_csv("Building_Structure.csv")
""" Add age_building , count_floors_pre_eq , count_floors_post_eq 
,height_ft_pre_eq , height_ft_post_eq,condition_post_eq

building_structure.head(15)
df = train.merge(building_structure,how="left",on=["building_id"])
df_test = test.merge(building_structure,how="left",on=["building_id"])
df.dtypes
"""
train.isnull().sum(axis=0)
train.fillna(-1,inplace = True)
test.fillna(-1,inplace=True)
#train.drop('building_id',axis=1,inplace=True)

id_ = []
for i in train['building_id'].values:
    s = str(i)
    int_ = int(s,16)
    id_.append(int_)
train['building_id'] = id_
id_ = []
for i in test['building_id'].values:
    s = str(i)
    int_ = int(s,16)
    id_.append(int_)
test['building_id'] = id_
# categorical encoding
train['damage_grade'].value_counts()
encode = { 'Grade 5':5, 'Grade 4':4,'Grade 3':3,'Grade 2':2,'Grade 1':1}
train['damage_grade'].replace(encode,inplace = True)
train['area_assesed'].value_counts()
encode_area ={ 'Not able to inspect' :0,'Interior':1,'Exterior':2,'Building removed':3,'Both':4}
train['area_assesed'].replace(encode_area,inplace=True)
test['area_assesed'].replace(encode_area,inplace=True)
target = train['damage_grade'].values
train.drop('damage_grade',axis=1,inplace=True)


train = pd.get_dummies(train,columns=['area_assesed'])
train.drop('area_assesed_0',axis=1,inplace=True)
test = pd.get_dummies(test,columns=['area_assesed'])
test.drop('area_assesed_0',axis=1,inplace=True)

from sklearn.ensemble import RandomForestClassifier
X = train.iloc[:,:].values
test_X = test.iloc[:,:].values
"""
clf = RandomForestClassifier(bootstrap=True, class_weight=None,
            criterion='entropy', max_depth=150, max_features='sqrt',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=2,
            min_samples_split=8, min_weight_fraction_leaf=0.0,
            n_estimators=200, n_jobs=-1, oob_score=True,
            random_state=None, verbose=5, warm_start=False)
clf.fit(X,target)
y_pred = clf.predict(test_X)
plt.plot(clf_.feature_importances_)
plt.xticks(np.arange(X.shape[1]),train.columns.tolist(),rotation=90)
"""

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,target,test_size=0.20,random_state=42)

clf_ = RandomForestClassifier(bootstrap=True, class_weight=None,
            criterion='entropy', max_depth=150, max_features='sqrt',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=2,
            min_samples_split=8, min_weight_fraction_leaf=0.0,
            n_estimators=200, n_jobs=-1, oob_score=True,
            random_state=None, verbose=5, warm_start=False)
clf_.fit(X_train,y_train)
y_pred = clf_.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cf = confusion_matrix(y_test,y_pred)
acu = accuracy_score(y_test,y_pred)

# submission
sub = pd.read_csv("sample_submission.csv")
sub['damage_grade'] = y_pred
encode_ = { 5:'Grade 5', 4:'Grade 4',3:'Grade 3',2:'Grade 2',1:'Grade 1'}
sub['damage_grade'].replace(encode_,inplace=True)
sub.to_csv("submission.csv",index=False)