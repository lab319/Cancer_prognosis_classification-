# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:34:07 2020

@author: Lab319
"""
import time
start = time.clock()
import numpy
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,Binarizer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import keras.backend as K
from tfdeterminism import patch
patch()

import tensorflow as tf
tf.random.set_seed(1024)
file_1=open("KIRP_methy.txt")
file_1.readline()
x=np.loadtxt(file_1)
file_1.close()

file_4=open("KIRP_label.txt")
file_4.readline()
y=np.loadtxt(file_4)
file_4.close()


seed = 7
numpy.random.seed(seed)

def create_model(neurons_1=25,neurons_2=5):
    # create model
    inputDims = x.shape[1]
    model = Sequential()
    model.add(Dense(neurons_1, input_dim=inputDims,init='uniform', activation='relu'))
    model.add(Dense(neurons_2,init='uniform',activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    optimizer = SGD(lr=0.1, momentum=0.8)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
    
scores_1=[]
scores_2=[]
scores_3=[]
scores_4=[]

i=1
kf = KFold(n_splits=10, shuffle=False, random_state=1)
for train_index,test_index in kf.split(x, y):
    x_train,x_test=x[train_index],x[test_index]
    y_train,y_test=y[train_index],y[test_index]
    
    ss=StandardScaler()
    x_train=ss.fit_transform(x_train)
    x_test=ss.transform(x_test)
 
    seed = 7
    numpy.random.seed(seed)
   
    model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=10)
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
    
    print ('AUC : %f' % auc)
    print ('ACC : %f' % acc) 
    print("PRAUC: %f" % pr_auc)
    print ('MCC : %f' % mcc)
    print ('SEN : %f' % sen)
    print ('SEP : %f' % sps)
    
    scores_1.append(auc)
    scores_2.append(acc)
    scores_3.append(pr_auc)
    scores_4.append(mcc) 
    i=i+1

    
print('auc-mean-score: %.3f' %np.mean(scores_1))
print('acc-mean-score: %.3f' %np.mean(scores_2))
print('pr-mean-score: %.3f' %np.mean(scores_3))
print('mcc-mean-score: %.3f' %np.mean(scores_4))

elapsed = (time.clock() - start)
print("Time used:",elapsed)







