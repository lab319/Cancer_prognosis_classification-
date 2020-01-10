# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:54:08 2019

@author: Lab319
"""

import numpy as np
from sklearn import metrics   #Additional     scklearn functions
from sklearn.preprocessing import StandardScaler   #Perforing grid search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

file_1=open("KIRP_methy.txt")
file_1.readline()
x=np.loadtxt(file_1)
file_1.close()

file_4=open("KIRP_label.txt")
file_4.readline()
y=np.loadtxt(file_4)
file_4.close()

scores_1=[]
scores_2=[]
scores_3=[]
scores_4=[]
i=1
kf = KFold(n_splits=10, shuffle=False, random_state=1)
for train_index,test_index in kf.split(x, y):
   print("i",i)
   x_train,x_test=x[train_index],x[test_index]
   y_train,y_test=y[train_index],y[test_index]
   
   #ss=StandardScaler()
   #x_train=ss.fit_transform(x_train)
   #x_test=ss.transform(x_test)
   
   param_grid = {"n_estimators": [100,500,1000,1500,2000,2500]}
   clf = RandomForestClassifier(random_state=0)
   
   kfold = KFold(n_splits=3, shuffle=False, random_state=7)
   grid = GridSearchCV(clf, param_grid, cv=kfold, scoring='roc_auc',n_jobs=-1,verbose=1)
   grid.fit(x_train, y_train)
   score_train=grid.best_score_
   print ("grid.best_score_",score_train)
   print ("grid.best_params_",grid.best_params_)
   
   best_params=grid.best_params_
   model=RandomForestClassifier(**best_params,random_state=0)
   
   model.fit(x_train, y_train)  
   y_proba=model.predict_proba(x_test)[:,1]
   y_pred = model.predict(x_test)
    
   auc = metrics.roc_auc_score(y_test, y_proba) 
   acc = accuracy_score(y_test, y_pred)
    
   precision, recall, _thresholds = metrics.precision_recall_curve(y_test, y_proba)
   pr_auc = metrics.auc(recall, precision)
   mcc = matthews_corrcoef(y_test, y_pred)
    
   tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
   total=tn+fp+fn+tp
   sen = float(tp)/float(tp+fn)
   sps = float(tn)/float((tn+fp))
   
   print("auc",auc)

   scores_1.append(auc)
   scores_2.append(acc)
   scores_3.append(pr_auc)
   scores_4.append(mcc)
   i=i+1

print('auc-mean-score: %.3f' %np.mean(scores_1))
print('acc-mean-score: %.3f' %np.mean(scores_2))
print('pr-mean-score: %.3f' %np.mean(scores_3))
print('mcc-mean-score: %.3f' %np.mean(scores_4))
