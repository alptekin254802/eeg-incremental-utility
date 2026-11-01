# The following script computes the nested CV accuracies
# for various parameter combinations in the training test.

import numpy as np
import pandas as pd
import scipy.stats
from sklearn import svm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from functions_open import *
import warnings

#ignore warnings-----------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning) 

#prepare training data---------------------------------------

#import csv file (modify according to your file path)
data0_pd=pd.read_csv(r"yourpath/MLBIAS_trainingSet_open.csv", sep=';', decimal=",") #load csv file as pandas data frame
#extract the 'Group' variable which acts as the label
data_labels_pd=data0_pd["Group"] 
#change labels to binary
data_labels=data_labels_pd.to_numpy() #convert to numpy array
data_labels[data_labels=="HC"]=0 #healthy subjects <= label 0
data_labels[data_labels=="ADHD_NoMed"]=1 #ADHD without meds <= label 1
data_labels=data_labels.astype(float) #convert to float array 
#remove the variables which are not required in training
data0_pd=data0_pd.drop(["Group"],axis=1)

shape =  data_labels.shape[0]
#choose a missing value handling method:
#     'impute' : KNN imputation
#     'zero'   : Zero imputation 
#     'mean'   : Mean value imputation
#     'remove' : Remove MV (not programmed, do not use)
mvMethod="impute"
# choose a rescale method:
#     'min-max' : tranform data to [0,1]
#     'zscore'  : z-score transform
rescale="min-max"
#pre-process data
data_pd=PreProcessData(data0_pd,mvMethod , rescale) #pre-process data 

#impute function changes the dataframe to an array, change it back
if mvMethod=="impute":
    data_pd.columns=list(data0_pd.columns)

clf=svm.SVC(kernel="linear",probability=1)

#choose scoring methods (check : https://scikit-learn.org/stable/modules/model_evaluation.html)
score= [accuracy_score, f1_score, sensitivity, specificity]

#create dataframes for all cases
dataframes=[]

numFeaturesMax=76

#create top parameters dataframes
top_params=createTopParamsDF(data_pd, data_labels,mrmr=1,n=numFeaturesMax)

for i in range(1,numFeaturesMax+1):
    top  = data_pd[top_params[0:i]]
    dataframes.append(top)
 
performanceParam=[]
scoreNames=[]

#store Nested CV accuracies when considering top params for all cases
for k in range(0,len(score)):
    scoreNames.append(score[k].__name__)
    nCV_acc=[]  
    for i in range(0,len(dataframes)):
        nCV_acc.append(nestedCrossValidation(dataframes[i],data_labels,clf,scoring=score[k],roc=1,k=5))
        #print('Case '+str(i)+':'+ str(nCV_acc[i]))
    performanceParam.append(nCV_acc)

performanceParam = pd.DataFrame(performanceParam).T   
# adding column name to the respective columns
performanceParam.columns = scoreNames


# Plot accuracy scores in separate plots------------------------------------------------
alpha = 0.05
n = 50
groups = 2
dim = np.arange(1, len(performanceParam["accuracy_score"])+1)

#accuracy
nCV_accuracy_line, = plt.plot(dim, performanceParam["accuracy_score"], color='#3d5941', marker='o', markersize=4.5)
plt.ylabel("Accuracy", fontsize="12")
plt.xlabel("Number of features", fontsize="12")
plt.xticks(dim)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ax = plt.gca()
temp = ax.xaxis.get_ticklabels()
temp = list(set(temp) - set(temp[::5]))
for label in temp:
    label.set_visible(False)
#plt.ylim(0.5, 1)
plt.margins(x=0.02)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([0.70, 0.95])
plt.rcParams['figure.dpi'] = 300
plt.show()

# sensitivity
nCV_sensitivity_line, = plt.plot(dim, performanceParam["sensitivity"], color='#ca562c', marker='o', markersize=4.5)
plt.ylabel("Sensitivity", fontsize="12")
plt.xlabel("Number of features", fontsize="12")
plt.xticks(dim)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ax = plt.gca()
temp = ax.xaxis.get_ticklabels()
temp = list(set(temp) - set(temp[::5]))
for label in temp:
    label.set_visible(False)
#plt.ylim(0.5, 1)
plt.margins(x=0.02)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([0.70, 0.95])
plt.rcParams['figure.dpi'] = 300
plt.show()

#specificity
nCV_specificity_line, = plt.plot(dim, performanceParam["specificity"], color='#d6ab45', marker='o', markersize=4.5)
plt.ylabel("Specificity", fontsize="12")
plt.xlabel("Number of features", fontsize="12")
plt.xticks(dim)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ax = plt.gca()
temp = ax.xaxis.get_ticklabels()
temp = list(set(temp) - set(temp[::5]))
for label in temp:
    label.set_visible(False)
#plt.ylim(0.5, 1)
plt.margins(x=0.02)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([0.70, 0.95])
plt.rcParams['figure.dpi'] = 300
plt.show()

