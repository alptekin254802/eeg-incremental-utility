# The following script computes the nested CV accuracies
# for various parameter combinations in the training test.

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import binom
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

#choose scoring method (check : https://scikit-learn.org/stable/modules/model_evaluation.html)
score= [accuracy_score, f1_score, sensitivity, specificity]

#create dataframes for all cases
dataframes=[]

optiNumFeatures=11
numFeaturesMax=76

#create top parameters dataframes
top_params=createTopParamsDF(data_pd, data_labels,mrmr=1,n=numFeaturesMax, scores=True)
 
for i in range(1,numFeaturesMax+1):
    top  = data_pd[top_params[0][0:i]]
    dataframes.append(top)
 
performanceParam=[]
scoreNames=[]


#train model with all data---------------------------------------------------
space=dict()
space["C"]=[0.1,1,10,100]
search = GridSearchCV(clf, space, scoring="accuracy", return_train_score=1)

#create models using params with highest CV accuracy
models=[]
for i in range(0,len(dataframes)):
    result=search.fit(dataframes[i],data_labels)
    models.append(result.best_estimator_)


#prepare test set---------------------------------------------------

#import csv file (modify according to your file path)
test0_pd=pd.read_csv(r"yourpath/MLBIAS_testSet_open.csv", sep=';', decimal=",") #load csv file as pandas data frame

#extract the "Group" variable which acts as the label
test_labels_pd=test0_pd["Group"] 
#change labels to binary
test_labels=test_labels_pd.to_numpy() #convert to numpy array
test_labels[test_labels=="HC"]=0 #healthy subjects <= label 0
test_labels[test_labels=="ADHD"]=1 #ADHD without meds <= label 1
test_labels=test_labels.astype(float) #convert to float array 
#remove the variables which are not required in training
test0_pd=test0_pd.drop(["Group"],axis=1)

#pre-process data
test_pd=PreProcessData(test0_pd,mvMethod , rescale) #pre-process data 

#impute function changes the dataframe to an array, change it back
if mvMethod=="impute":
    test_pd.columns=list(test0_pd.columns)


#create testing dataframes for all cases
testframes=[]

for i in range(0,numFeaturesMax):
    testframes.append(test_pd[dataframes[i].columns])


# specify confidence interval
confidence = 0.95
z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)

#check classifier accuracy 

test_performance=[]
test_conf=[]
scoreNames=[]

for k in range(0,len(score)):
    scoreNames.append(score[k].__name__)
    test_acc=[]
    conf=[]
    for i in range(0,len(models)):
        testframes[i]=reorderColumns(dataframes[i], testframes[i])
        test_predict=models[i].predict(testframes[i])
        test_acc.append(round(score[k](test_labels,test_predict),2))
        # confidence interval for test accuracy
        ci_length = z_value*np.sqrt((score[k](test_labels,test_predict)*(1 - score[k](test_labels,test_predict))) / test_labels.shape[0])
        conf.append(ci_length)
    test_performance.append(test_acc)
    test_conf.append(conf)
    
test_performance = pd.DataFrame(test_performance).T
test_conf = pd.DataFrame(test_conf).T   
# adding column name to the respective columns
test_performance.columns = scoreNames
test_conf.columns = scoreNames

#Plot test performance scores as subplots----------------------------------------------------------------------------
alpha = 0.05
n = 36
groups = 2
signLevel = binom.ppf(1-alpha, n, 1/groups)*100/n  # calculate theoretical significance level

dim = np.arange(1, len(test_performance["accuracy_score"])+1)

#accuracy
test_accuracy_line, = plt.plot(dim, test_performance["accuracy_score"], color='#3d5941', marker='o',markersize=4.5)
plt.fill_between(dim, (test_performance["accuracy_score"]-test_conf["accuracy_score"]), (test_performance["accuracy_score"]+test_conf["accuracy_score"]), color = '#3d5941', alpha=.2)
plt.axhline(y = signLevel/100, color = '0', linestyle = '-') 
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
plt.margins(x=0.02)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([0.35, 1])
plt.rcParams['figure.dpi'] = 300
plt.show()

#sensitivity
test_sensitivity_line, = plt.plot(dim, test_performance["sensitivity"], color='#ca562c', marker='o', markersize=4.5)
plt.fill_between(dim, (test_performance["sensitivity"]-test_conf["sensitivity"]), (test_performance["sensitivity"]+test_conf["sensitivity"]), color = '#ca562c', alpha=.2)
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
plt.margins(x=0.02)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([0.35, 1])
plt.rcParams['figure.dpi'] = 300
plt.show()

#specificity
test_specificity_line, = plt.plot(dim, test_performance["specificity"], color='#d6ab45', marker='o',markersize=4.5)
plt.fill_between(dim, (test_performance["specificity"]-test_conf["specificity"]), (test_performance["specificity"]+test_conf["specificity"]), color = '#d6ab45', alpha=.2)
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
plt.margins(x=0.02)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([0.35, 1])
plt.rcParams['figure.dpi'] = 300
plt.show()



#Plot Top Parameters------------------------------------------------------------------------------
top_relevance_scores = top_params[1][top_params[0]]
top11_relevance = top_relevance_scores[0:11]

top11_rel = top11_relevance.to_frame()
top11_rel.rename( columns={0 :'rel'}, inplace=True )

top11_rel['name'] = top11_rel.index
top11_rel = top11_rel.assign(group=top11_rel['name'].str[:3])
top11_rel['name_new'] = ['(1) Inattention', '(2) Avg Saccade Duration (NDP)', '(3) CV Fixation Duration (NDP)', '(4) Bias (DP)', '(5) Hyperactivity', '(6) Impulsivity', '(7) CV Saccade Peak Velocity (NDP)', '(8) CV Reaction Time (NDP)', '(9) Head Position', '(10) CV Fixation Duration (DP)', '(11) dPrime (DP)']

top11_rel = top11_rel .iloc[::-1]
top11_rel.rel = top11_rel.rel.round(1)

 
my_color=np.where(top11_rel['group']=='CPT', '#3d5941', np.where(top11_rel['group']=='ET_', '#ca562c', np.where(top11_rel['group']=='ES_', '#d6ab45', '#80cdc1')))
my_range=range(1,len(top11_rel.index)+1)
plt.figure(figsize=(20,10)) 
plt.hlines(y=my_range, xmin=0, xmax=top11_rel['rel'], color=my_color, alpha=0.4)
plt.scatter(top11_rel['rel'], my_range, color=my_color, alpha=1, s = 500)

for i, txt in enumerate(top11_rel.rel):
    plt.annotate(txt, (top11_rel['rel'][i], my_range[i]), horizontalalignment='center', verticalalignment='center', color = 'white')
 

plt.yticks(my_range, top11_rel['name_new'])
plt.margins(x=0)
ax = plt.gca()
ax.set_xlim([-1, 60])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.rcParams['figure.dpi'] = 300
plt.show()


#Plot pie chart------------------------------------------------------------------------------------
labels = 'Experience sampling', 'Eye tracking', 'CPT performance', 'Actigraphy'
sizes = [27, 37, 27, 9]
color = ['#d6ab45', '#ca562c', '#3d5941', '#80cdc1']


plt.pie(sizes, autopct='%.0f%%', colors=color)
plt.legend(labels, bbox_to_anchor=(0.9,1.025))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75)
plt.rcParams['figure.dpi'] = 300
plt.show()

















