
# coding: utf-8

# In[9]:


import xgboost
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import sklearn.metrics 

class GV:
    '''
    Scoring	Function	   Comment
    *Classification
    ‘accuracy’             metrics.accuracy_score
    ‘average_precision’	   metrics.average_precision_score
    ‘f1’	               metrics.f1_score	for binary targets
    ‘f1_micro’	           metrics.f1_score	micro-averaged
    ‘f1_macro’         	   metrics.f1_score	macro-averaged
    ‘f1_weighted’	       metrics.f1_score	weighted average
    ‘f1_samples’	       metrics.f1_score	by multilabel sample
    ‘neg_log_loss’	       metrics.log_loss	requires predict_proba support
    ‘precision’ etc.	   metrics.precision_score	suffixes apply as with ‘f1’
    ‘recall’ etc.	       metrics.recall_score	suffixes apply as with ‘f1’
    ‘roc_auc’	           metrics.roc_auc_score

    *Clustering
    ‘adjusted_rand_score’	metrics.adjusted_rand_score

    *Regression
    ‘neg_mean_absolute_error’	metrics.mean_absolute_error
    ‘neg_mean_squared_error’	metrics.mean_squared_error
    ‘neg_median_absolute_error’	metrics.median_absolute_error
    ‘r2’	metrics.r2_score
    '''

    def xg_find_base(self, scoring, data_x, data_y, model_xg, params, overfit=None):
        kfold = KFold(n_splits=3, shuffle=False, random_state=7)
        params = {}
        params_test1 = {"max_depth": np.arange(3, 8, 1)}
        clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=-1, scoring=scoring)
        clf.fit(data_x, data_y)
        params.update({'max_depth': clf.best_params_["max_depth"]})
        model_xg.max_depth = clf.best_params_["max_depth"]
        print(clf.best_params_)
        print("clf.best_score_", clf.best_score_)

#         params_test1 = {"n_estimators": np.arange(30, 100, 10), 'learning_rate': np.arange(0.01, 0.1, 0.01)}
#         clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=1, scoring=scoring)
#         clf.fit(data_x, data_y)
#         params.update({'learning_rate': clf.best_params_["learning_rate"]})
#         params.update({'n_estimators': clf.best_params_["n_estimators"]})
#         model_xg.learning_rate = clf.best_params_["learning_rate"]
#         model_xg.n_estimators = clf.best_params_["n_estimators"]
#         print(clf.best_params_)
#         print("clf.best_score_", clf.best_score_)

        params_test1 = {"learning_rate": [0.005,0.01,0.03,0.05,0.07,0.1,0.13,0.15,0.2]}
        clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=-1, scoring=scoring)
        clf.fit(data_x, data_y)
        params.update({'learning_rate': clf.best_params_["learning_rate"]})
        model_xg.learning_rate = clf.best_params_["learning_rate"]
        print(clf.best_params_)
        print("clf.best_score_", clf.best_score_)
        
        params_test1 = {"colsample_bytree": np.arange(0.3, 0.9, 0.1), 'subsample': np.arange(0.3, 0.9, 0.1)}
        clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=-1, scoring=scoring)
        clf.fit(data_x, data_y)
        params.update({'colsample_bytree': clf.best_params_["colsample_bytree"]})
        params.update({'subsample': clf.best_params_["subsample"]})
        model_xg.colsample_bytree = clf.best_params_["colsample_bytree"]
        model_xg.subsample = clf.best_params_["subsample"]
        print(clf.best_params_)
        print("clf.best_score_", clf.best_score_)

        params_test1 = {"gamma": np.arange(0, 1.2, 0.1)}
        clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=-1, scoring=scoring)
        clf.fit(data_x, data_y)
        params.update({'gamma': clf.best_params_["gamma"]})
        model_xg.gamma = clf.best_params_["gamma"]
        print(clf.best_params_)
        print("clf.best_score_", clf.best_score_)
        
        params_test1 = {"colsample_bylevel": np.arange(0.3, 0.9, 0.1), "colsample_bynode": np.arange(0.4, 1, 0.1)}
        #params_test1 = {"colsample_bylevel": np.arange(0.3, 1.0, 0.1), "colsample_bynode": np.arange(0.3, 1.0, 0.1)}
        clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=-1, scoring=scoring)
        clf.fit(data_x, data_y)
        params.update({'colsample_bylevel': clf.best_params_["colsample_bylevel"]})
        params.update({'colsample_bynode': clf.best_params_["colsample_bynode"]})
        model_xg.colsample_bylevel = clf.best_params_["colsample_bylevel"]
        model_xg.colsample_bynode = clf.best_params_["colsample_bynode"]
        print(clf.best_params_)
        print("clf.best_score_", clf.best_score_)

        params_test1 = {'reg_lambda': np.arange(1, 1.6, 0.1), 'reg_alpha': np.arange(0, 1.6, 0.1)}
        clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=-1, scoring=scoring)
        clf.fit(data_x, data_y)
        params.update({'reg_alpha': clf.best_params_["reg_alpha"]})
        params.update({'reg_lambda': clf.best_params_["reg_lambda"]})
        model_xg.reg_lambda = clf.best_params_["reg_lambda"]
        model_xg.reg_alpha = clf.best_params_["reg_alpha"]
        print(clf.best_params_)
        print("clf.best_score_", clf.best_score_)

        return model_xg, params

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,Binarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
scores_1=[]
scores_2=[]
scores_3=[]
scores_4=[]
scores_5=[]
scores_6=[]
data = pd.read_table("HNSC_mrna.txt")
data.columns = [str(i) for i in range(data.shape[1])]

label = pd.read_table("HNSC_label.txt").values.reshape([-1, ])

kf = KFold(n_splits=10, shuffle=False, random_state=1)
y_valid_pred_total = np.zeros(data.shape[0])
score = []
para=[]
print(data.shape, label.shape)
for train_ind, test_ind in kf.split(data, label):
    train_data = data.iloc[train_ind, :]
    train_y = label[train_ind]
    test_data = data.iloc[test_ind, :]
    test_y = label[test_ind]
    model = XGBClassifier()
    gv = GV()
    params = {}

    model, params = gv.xg_find_base('roc_auc', train_data, train_y, model, {})
    #     model,params = gv.xg_find_up('roc_auc',train_data,train_y,model,{},overfit=True)
    print(params)
    early_stop = 50
    verbose_eval = 0
    num_rounds = 450


    d_train = xgb.DMatrix(train_data, label=train_y)
    d_valid = xgb.DMatrix(test_data, label=test_y)
    watchlist = [(d_train, 'train')]
    params.update({'eval_metric': 'auc',
                   'objective': 'binary:logistic'})
    model = xgb.train(params, d_train, num_boost_round=num_rounds, early_stopping_rounds=early_stop, evals=watchlist)

    y_pre = model.predict(d_valid).reshape([-1,1])
    mms = Binarizer(0.5)
    y_pre_ = mms.fit_transform(y_pre)
    auc = metrics.roc_auc_score(test_y, y_pre) 
    acc = accuracy_score(test_y, y_pre_)
    
    precision, recall, _thresholds = metrics.precision_recall_curve(test_y, y_pre_)
    pr_auc = metrics.auc(recall, precision)
    mcc = matthews_corrcoef(test_y, y_pre_)
    
    tn, fp, fn, tp = confusion_matrix(test_y, y_pre_).ravel()
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
    scores_5.append(sen)
    scores_6.append(sps)
    para.append(params)

    
print('auc-mean-score: %.3f' %np.mean(scores_1))
print('acc-mean-score: %.3f' %np.mean(scores_2))
print('pr-mean-score: %.3f' %np.mean(scores_3))
print('mcc-mean-score: %.3f' %np.mean(scores_4))
print('sen-mean-score: %.3f' %np.mean(scores_5))
print('sps-mean-score: %.3f' %np.mean(scores_6))
print(para)

'''
auc-mean-score: 0.602
acc-mean-score: 0.794
pr-mean-score: 0.438
mcc-mean-score: 0.085
sen-mean-score: 0.055
sps-mean-score: 0.987

[{'max_depth': 7, 'learning_rate': 0.13, 'colsample_bytree': 0.9000000000000001, 'subsample': 0.4, 'gamma': 0.0, 'colsample_bylevel': 0.3, 'colsample_bynode': 0.5, 'reg_alpha': 0.1, 'reg_lambda': 1.2000000000000002, 'eval_metric': 'auc', 'objective': 'binary:logistic'}, {'max_depth': 3, 'learning_rate': 0.15, 'colsample_bytree': 0.9000000000000001, 'subsample': 0.6000000000000001, 'gamma': 1.1, 'colsample_bylevel': 0.7000000000000002, 'colsample_bynode': 0.7, 'reg_alpha': 1.1, 'reg_lambda': 1.5000000000000004, 'eval_metric': 'auc', 'objective': 'binary:logistic'}, {'max_depth': 3, 'learning_rate': 0.13, 'colsample_bytree': 0.8000000000000003, 'subsample': 0.3, 'gamma': 0.7000000000000001, 'colsample_bylevel': 0.9000000000000001, 'colsample_bynode': 0.8999999999999999, 'reg_alpha': 0.0, 'reg_lambda': 1.6000000000000005, 'eval_metric': 'auc', 'objective': 'binary:logistic'}, {'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.7000000000000002, 'subsample': 0.9000000000000001, 'gamma': 0.0, 'colsample_bylevel': 0.9000000000000001, 'colsample_bynode': 0.7999999999999999, 'reg_alpha': 0.0, 'reg_lambda': 1.0, 'eval_metric': 'auc', 'objective': 'binary:logistic'}, {'max_depth': 5, 'learning_rate': 0.13, 'colsample_bytree': 0.5, 'subsample': 0.4, 'gamma': 0.1, 'colsample_bylevel': 0.9000000000000001, 'colsample_bynode': 0.5, 'reg_alpha': 0.0, 'reg_lambda': 1.6000000000000005, 'eval_metric': 'auc', 'objective': 'binary:logistic'}, {'max_depth': 6, 'learning_rate': 0.13, 'colsample_bytree': 0.8000000000000003, 'subsample': 0.7000000000000002, 'gamma': 0.0, 'colsample_bylevel': 0.6000000000000001, 'colsample_bynode': 0.5, 'reg_alpha': 0.4, 'reg_lambda': 1.1, 'eval_metric': 'auc', 'objective': 'binary:logistic'}, {'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.4, 'subsample': 0.8000000000000003, 'gamma': 0.0, 'colsample_bylevel': 0.8000000000000003, 'colsample_bynode': 0.7999999999999999, 'reg_alpha': 0.6000000000000001, 'reg_lambda': 1.2000000000000002, 'eval_metric': 'auc', 'objective': 'binary:logistic'}, {'max_depth': 3, 'learning_rate': 0.03, 'colsample_bytree': 0.8000000000000003, 'subsample': 0.8000000000000003, 'gamma': 1.0, 'colsample_bylevel': 0.9000000000000001, 'colsample_bynode': 0.5, 'reg_alpha': 0.5, 'reg_lambda': 1.0, 'eval_metric': 'auc', 'objective': 'binary:logistic'}, {'max_depth': 4, 'learning_rate': 0.13, 'colsample_bytree': 0.9000000000000001, 'subsample': 0.4, 'gamma': 1.1, 'colsample_bylevel': 0.3, 'colsample_bynode': 0.7999999999999999, 'reg_alpha': 1.4000000000000001, 'reg_lambda': 1.2000000000000002, 'eval_metric': 'auc', 'objective': 'binary:logistic'}, {'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.3, 'subsample': 0.9000000000000001, 'gamma': 0.0, 'colsample_bylevel': 0.7000000000000002, 'colsample_bynode': 0.7, 'reg_alpha': 0.6000000000000001, 'reg_lambda': 1.2000000000000002, 'eval_metric': 'auc', 'objective': 'binary:logistic'}]
'''