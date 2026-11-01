# The following script contains the functions
# necessary for conducting the machine learning
# based analysis for ADHD diagnosis based on 
# a VR experiment.

import numpy as np
import pandas as pd
import scipy.stats as stats
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from mrmr import mrmr_classif    #(see https://github.com/smazzanti/mrmr)

#data must be a pandas dataframe
def missingVals(data,method):
    if method=="remove":
        return data.dropna()
    if method=="zero":
        return data.fillna(0)
    if method=="mean":
        return data.fillna(data.mean())
    if method=="impute":
        #imp=IterativeImputer(max_iter=100,verbose=1,random_state=0,n_nearest_features=10)
        imp=KNNImputer(n_neighbors=10)
        imputedData= imp.fit_transform(data)
        return imputedData
     
#returns the columns with outliers. One may plot the values in the columns in separate plots if 'plot' is set to 1
#Note that outliers in a column will be values whose z-score is <3 or >3
def getOutliers(data, plot=0):
    data_zscore_abs=stats.zscore(data).abs()
    outlierCols=[]     
    
    outliersExit=lambda col: 1 if data_zscore_abs[data_zscore_abs[col]>=3].size>=1  else 0
    
    for col in data_zscore_abs:
        if outliersExit(col):
            outlierCols.append(col)
    
    if plot==1:
        for i in outlierCols:
            fig,ax=plt.subplots()
            ax.hist(data[i])
        
    return outlierCols

def rescaleData(data,method="min-max"):
    if method=="min-max":
        return (data-data.min())/(data.max()-data.min())
    if method=="zscore":
        return stats.zscore(data)

#Pre-proesses data by:
    # -dealing with missing values
    # -rescaling data
def PreProcessData(data,mvMethod,scaling):    
    data_mv=missingVals(data,mvMethod)
    data_mv_pd=pd.DataFrame(data_mv)
    data_mv_rescaled=rescaleData(data_mv_pd,scaling)
    return data_mv_rescaled   

#creates a dataframe consisting of only the top parameters based on MRMR criterion
def createTopParamsDF(dataframe,labels,mrmr=None, n=20,k=5, scores=False):
    assert mrmr is not None
    if mrmr is not None:
        if not scores:  # if scores = false, only return list of top  params
            topParams = mrmr_classif(X=dataframe, y=labels, K=n)
            return topParams
        if scores:  # if scores = true, return top params + mrmr scores for all params
            topParams = mrmr_classif(X=dataframe, y=labels, K=n, return_scores = True)
            return topParams
        

def sensitivity(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp+fn)

def specificity(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn+fp)

# nested cross validation
def nestedCrossValidation(dataframe, labels, classifier, k=5,scoring=accuracy_score,roc=0):
    if roc==1:
        roc_auc=[]
        
    cv_outer=KFold(n_splits=k, shuffle=True, random_state=1)
    outer_results=list()
    
    for train_idx,test_idx in cv_outer.split(dataframe):
        X_train, X_test=dataframe.iloc[train_idx],dataframe.iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        
        space=dict()
        space["C"]=[0.1,1,10,100]
        #space["gamma"]=[1,0.1,0.01,0.001]
        
        
        clf=classifier
        search = GridSearchCV(clf, space, scoring="accuracy", cv=cv_inner, refit=True, return_train_score=1)
        result=search.fit(X_train,y_train)

        #basing outer fold selection on max. accuracy---------------------
        y_pred=result.best_estimator_.predict(X_test)
        #-----------------------------------------------------------------
        
        acc=scoring(y_test,y_pred)
       
        outer_results.append(acc)
        
        if roc==1:
            Y_score=result.predict_proba(X_test)[:,1]
            roc_auc.append(roc_auc_score(y_test,Y_score))

        
    return statistics.mean(outer_results)

#reorders the columns of dataframe 'target' based on 'reference'
def reorderColumns(ref,target):
    refCols=ref.columns
    targetCols=target.columns
    assert refCols.size==targetCols.size
    for i in range(0,targetCols.size):
        assert targetCols[i] in refCols and refCols[i] in targetCols

    return target.reindex(columns=refCols)