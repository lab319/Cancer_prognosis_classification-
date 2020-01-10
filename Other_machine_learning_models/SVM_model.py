# -*- coding: utf-8 -*-
"""
Created on Mon Dec 02 15:56:29 2019

@author: Administrator
"""

import numpy as np
from sklearn import metrics    
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
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
scores_5=[]
scores_6=[]
i=1
kf = KFold(n_splits=10, shuffle=False, random_state=1)
for train_index,test_index in kf.split(x, y):
   print("i",i)
   x_train,x_test=x[train_index],x[test_index]
   y_train,y_test=y[train_index],y[test_index]
   
   param_grid = [{'kernel': ['linear'],'C': [1e-2,1e-3,0.1, 1.0, 10,1e2,1e3,1e4,1e5,1e6,1e7,1e8]}]
   clf = SVC(probability=True)
   
   kfold = KFold(n_splits=3, shuffle=False, random_state=7)
   grid = GridSearchCV(clf, param_grid, cv=kfold, scoring='roc_auc',n_jobs=-1,verbose=1)
   grid.fit(x_train, y_train)
   score_train=grid.best_score_
   print ("grid.best_score_",score_train)
   print ("grid.best_params_",grid.best_params_)
   
   best_params=grid.best_params_
   model=SVC(**best_params,probability=True)
   
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
   scores_5.append(sen)
   scores_6.append(sps)
   i=i+1

print('auc-mean-score: %.3f' %np.mean(scores_1))
print('acc-mean-score: %.3f' %np.mean(scores_2))
print('pr-mean-score: %.3f' %np.mean(scores_3))
print('mcc-mean-score: %.3f' %np.mean(scores_4))

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   