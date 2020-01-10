# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:54:22 2019

@author: lab319
"""

import tensorflow as tf
from numpy.random import seed 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import  metrics   
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense,Activation,Input,Dropout
from keras.models import Sequential,Model
from keras.optimizers import SGD, Adadelta, Adagrad
import keras as K
import pickle
from sklearn.metrics import  roc_auc_score   
from sklearn.feature_selection import SelectPercentile, f_classif
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess =tf.compat.v1.Session(config=config)


file_1=open("KIRP_methy.txt")
file_1.readline()
x_methy=np.loadtxt(file_1)
file_1.close()

file_1=open("KIRP_mRNA.txt")
file_1.readline()
x_mRNA=np.loadtxt(file_1)
file_1.close()

file_4=open("KIRP_label.txt")
file_4.readline()
y=np.loadtxt(file_4)
file_4.close()

ss=StandardScaler()
x_methy=ss.fit_transform(x_methy)
x_mRNA=ss.fit_transform(x_mRNA)
x=np.column_stack((x_methy,x_mRNA))

i=0
scores_1=[]
scores_2=[]
scores_3=[]
scores_4=[]

skf = StratifiedKFold(n_splits=10,random_state=43,shuffle=False)
for train_index, test_index in skf.split(x, y):
    seed(1)
    tf.random.set_seed(7)
    x_methy_train,x_methy_test= x_methy[train_index], x_methy[test_index]
    x_mRNA_train,x_mRNA_test= x_mRNA[train_index], x_mRNA[test_index]
    y_train,y_test=y[train_index],y[test_index]
    
   
#    inputDims = x_mRNA_train.shape[1]

#    EncoderDims = 256
#    EncoderDims2 = 124

#    AutoEncoder = Sequential()
#    AutoEncoder.add(Dense(input_dim=inputDims,output_dim=EncoderDims,activation='sigmoid',name="Dense_1"))
#    AutoEncoder.add(Dense(input_dim=EncoderDims,output_dim=EncoderDims2,activation='sigmoid',name="Dense_2"))
#    AutoEncoder.add(Dense(input_dim=EncoderDims2,output_dim=EncoderDims,activation='sigmoid',name="Dense_3"))
#    AutoEncoder.add(Dense(input_dim=EncoderDims,output_dim=inputDims,activation='tanh',name="Dense_4"))
    
#    AutoEncoder.compile(optimizer='Adadelta',loss='binary_crossentropy')
#    AutoEncoder.fit(x_mRNA_train,x_mRNA_train,batch_size=16,nb_epoch=100,shuffle=False) 
    
#    pickle.dump(AutoEncoder,open("4-"+str(i)+".pkl","wb"))
    origin_model = pickle.load(open("KIRP-"+str(i)+".pkl","rb"))
    
    dense_layer_model = Model(inputs=origin_model.input,outputs=origin_model.get_layer('Dense_4').output)
    
    x_mRNA_train_feature = dense_layer_model.predict(x_mRNA_train)
    x_mRNA_test_feature = dense_layer_model.predict(x_mRNA_test)
    
    train_feature=np.column_stack((x_methy_train,x_mRNA_train_feature))
    test_feature=np.column_stack((x_methy_test,x_mRNA_test_feature))
    
    model = XGBClassifier(nthread=6)
    model.fit(train_feature, y_train)  
    y_proba=model.predict_proba(test_feature)[:,1]
    y_pred = model.predict(test_feature)
    	
    auc = metrics.roc_auc_score(y_test, y_proba) 
    acc = accuracy_score(y_test, y_pred)
    
    precision, recall, _thresholds = metrics.precision_recall_curve(y_test, y_proba)
    pr_auc = metrics.auc(recall, precision)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total=tn+fp+fn+tp
    sen = float(tp)/float(tp+fn)
    sps = float(tn)/float((tn+fp))
    
    print ('AUC Score (Test): %f' % auc)
    print ('ACC Score (Test): %f' % acc) 
    print ("PRAUC: %1.3f" % pr_auc)
    print ('MCC : %f' % mcc)
    
    scores_1.append(auc)
    scores_2.append(acc)
    scores_3.append(pr_auc)
    scores_4.append(mcc)
    i=i+1

print('auc-mean-score: %.3f' %np.mean(scores_1))

print('acc-mean-score: %.3f' %np.mean(scores_2))

print('pr-mean-score: %.3f' %np.mean(scores_3))

print('mcc-mean-score: %.3f' %np.mean(scores_4))
 

