#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:37:12 2019

@author: Mojtaba Jafaritadi
"""
import time
start = time.time()

from HeartBeatDetector import *
from HRV_parameters import *
from complexity_estimators import *
from autoregressive_model import *
from afib_detector import *
from h5_helperfunctions import *
from machine_learning_tools import *
import matplotlib.pyplot as plt
import pandas as pd

MODE_AF_signals=load_dict_from_hdf5('MODE_AF_rawsignals.h5')
UniHill_signals=load_dict_from_hdf5('UniHill_rawsignals.h5')

y_modeaf=pd.read_hdf('labels_modeaf.h5','y_train')
Y_modeaf=np.array(y_modeaf['y_train'])

y_unihill=pd.read_hdf('labels_unihillaf.h5','y_test')
Y_unihill=np.array(y_unihill['y_test'])

list_of_key_MODEAFnames=list(MODE_AF_signals.keys())     
list_of_key_UniHillnames=list(UniHill_signals.keys())     


############################################### First dataset (300 MODEAF (AFib=150))
signals_fused=dict()
heart_beat_locs=dict()
HRV_features=dict()
AFib_autocorr=dict()
Complexity_features=dict()
AR_features=dict()

fs=200 #Hz

for i in list_of_key_MODEAFnames[6:7]:

    
    plt.close('all')
    
####################################### Detecting Heartrates and extracting features 
    data=preprocess_resample(MODE_AF_signals[i]['acc'], MODE_AF_signals[i]['gyro'])
    
    fused_signal,Peak_locs= heartbeatdetector(data)
    signals_fused[i]=fused_signal
    heart_beat_locs[i]=Peak_locs
    
    time_domain_features=get_hrv_features(Peak_locs,fs)
    HRV_features[i]=time_domain_features
    
    entropy_features= get_complexity_features(fused_signal, fs)
    Complexity_features[i]=entropy_features
    
    estimated_variance= get_autoregressive_feature(Peak_locs,fs)
    AR_features[i]=estimated_variance
    
    AFib_autocorr[i]=detect_period(data, fs)

#######################################################Stacking all extracted features (MODEAF)
    
HRV_feat_modeaf=pd.DataFrame.from_dict(HRV_features, orient='index') ## transform HRV features to list
ACR_feat_modeaf=pd.DataFrame.from_dict(AFib_autocorr, orient='index')# transform Autocorrelatio features to list
CPLX_feat_modeaf=pd.DataFrame.from_dict(Complexity_features, orient='index')# transform Autocorrelatio features to list
AR_feature_modeaf=pd.DataFrame.from_dict(AR_features, orient='index')# transform autoregressive features to list
all_features_modeaf=pd.concat([HRV_feat_modeaf,CPLX_feat_modeaf,AR_feature_modeaf],axis=1)
all_features_modeaf['reg_index'] = ACR_feat_modeaf['regularity_index'] # add  regularity index feature to HRV features

all_features_modeaf['labels']=Y_modeaf
ACR_pred_modeaf=ACR_feat_modeaf['afibval']

################################################## second dataset (122 UniHill(n=85)+TYKS AFibs (n=37))
signals_fused=dict()
heart_beat_locs=dict()
HRV_features=dict()
AFib_autocorr=dict()
Complexity_features=dict()
AR_features=dict()

fs=200 #Hz

for i in list_of_key_UniHillnames[:]:

    
    plt.close('all')
    
####################################### Detecting Heartrates and extracting features
    data=preprocess_resample(UniHill_signals[i]['acc'], UniHill_signals[i]['gyro'])
    
    fused_signal,Peak_locs= heartbeatdetector(data)
    signals_fused[i]=fused_signal
    heart_beat_locs[i]=Peak_locs
    
    time_domain_features=get_hrv_features(Peak_locs,fs)
    HRV_features[i]=time_domain_features
    
    entropy_features= get_complexity_features(fused_signal, fs)
    Complexity_features[i]=entropy_features
    
    estimated_variance= get_autoregressive_feature(Peak_locs,fs)
    AR_features[i]=estimated_variance
    
    AFib_autocorr[i]=detect_period(data, fs)

#######################################################Stacking all extracted features (unihill+tyks afib)

HRV_feat_unihill=pd.DataFrame.from_dict(HRV_features, orient='index') ## transform HRV features to list
ACR_feat_unihill=pd.DataFrame.from_dict(AFib_autocorr, orient='index')# transform Autocorrelatio features to list
CPLX_feat_unihill=pd.DataFrame.from_dict(Complexity_features, orient='index')# transform Autocorrelatio features to list
AR_feature_unihill=pd.DataFrame.from_dict(AR_features, orient='index')# transform autoregressive features to list
all_features_unihill=pd.concat([HRV_feat_unihill,CPLX_feat_unihill,AR_feature_unihill],axis=1)
all_features_unihill['reg_index'] = ACR_feat_unihill['regularity_index'] # add  regularity index feature to HRV features
all_features_unihill['labels']=Y_unihill

ACR_pred_unihill=ACR_feat_unihill['afibval']

#####################################################
full_dataset_smartphone= pd.concat([all_features_modeaf,all_features_unihill],ignore_index=True)
all_features_modeaf.to_hdf('dataset_afib_classification_modeaf.h5', key='dataset', mode='w')
all_features_unihill.to_hdf('dataset_afib_classification_unihill.h5', key='dataset', mode='w')
full_dataset_smartphone.to_hdf('dataset_afib_classification_422subjects.h5', key='dataset', mode='w')

all_true_labels=np.concatenate((np.array(Y_modeaf),np.array(Y_unihill)))
all_acr_preds=np.concatenate((np.array(ACR_pred_modeaf),np.array(ACR_pred_unihill)))





##################################################### Autocorrelation evaluation
plt.figure()
plt.close('all')

np.set_printoptions(precision=2)
class_names=np.array(['Sinus','AFib'])
plot_confusion_matrix(all_true_labels, all_acr_preds, classes=class_names, title='Confusion matrix XCorr, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(all_true_labels, all_acr_preds, classes=class_names, normalize=True,title='Confusion matrix XCorr,Normalized confusion matrix')
plt.show()

######################################### ML PART
################################################# Import Libraries
from matplotlib.colors import ListedColormap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

############################################################### Initializing the classifiers/models

model_RF = RandomForestClassifier(n_estimators=250,random_state=1368) #define random forest as a fitting model to classify AFib
model_LR= LogisticRegression() #define logistic regression  as the fitting model to classify AFib
model_GB = GradientBoostingClassifier(n_estimators=34, learning_rate=1.0, max_depth=1, random_state=0)
model_AB = AdaBoostClassifier(n_estimators=20)
model_GB = GradientBoostingClassifier(n_estimators=64, learning_rate=0.007, max_depth=1, random_state=0)
model_NN =MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(10,5),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=100, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5,  random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)


####################################################### Shuffling and 

full_dataset_shuffled = shuffle(full_dataset_smartphone)

################################################# Primary leave one out CV on all data
sc = StandardScaler()

loocv = model_selection.LeaveOneOut() 
X_all_norm=sc.fit_transform(full_dataset_shuffled.loc[:, 'mean_nni':'reg_index'])
y_all=full_dataset_shuffled['labels']




########################################## Classifiaction part


############################################ Training different classifiers for LOOCV
clf1 = model_AB
clf2 = model_GB
clf3 = model_RF
clf4 = model_NN
clf5 = GaussianNB()

eclf = VotingClassifier(estimators=[('ab', clf1), ('gb', clf2), ('rf', clf3), ('lr', clf4), ('gnb', clf5)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Adaptive Boosting', 'Gradient Boosting', 'Random Forest', 'Neural Network','Naive Bayes','Ensemble']):
    scores = cross_val_score(clf, X_all_norm,y_all, cv=loocv, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
# 
    
    
############splitting the data into train and test (randomly)
dict_of_results = {}
for index in range(200): 
    tempDict = {}
    iterations = 'iteration' + str(index)
    clf1 = model_AB
    clf2 = model_GB
    clf3 = model_RF
    clf4 = model_NN
    clf5 = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(full_dataset_shuffled.loc[:, 'rmssd':'reg_index'], full_dataset_shuffled['labels'], test_size=0.3, random_state=np.random.randint(low=1, high=1000))
    
    X_train_norm = sc.fit_transform(X_train)
    X_test_norm = sc.transform (X_test)
       
    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Adaptive Boosting', 'Gradient Boosting', 'Random Forest', 'Neural Network','Naive Bayes','Ensemble']):
        model=clf.fit(X_train_norm, y_train)
        scores = model.score(X_test_norm, y_test)
#        values = [scores.mean(), scores.std()]
        tempDict[label] = scores.mean()    
        dict_of_results[iterations] = tempDict
        print("F1(micro): %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        
results_iterated=pd.DataFrame.from_dict(dict_of_results, orient='index') ## transform results to df
results_iterated.index = pd.RangeIndex(len(results_iterated.index))
results_iterated.plot()
plt.ylabel('accuracy (%)')

time_series_df = results_iterated['Ensemble']
smooth_path = time_series_df.rolling(3).mean()
path_deviation = time_series_df.rolling(3).std() 
plt.plot(smooth_path, linewidth=2)
plt.fill_between(results_iterated.index,(smooth_path-path_deviation), (smooth_path+path_deviation), color='c', alpha=.1)
plt.show()


#####
all_features_unihill.reset_index(inplace=True, drop = True)
ar_afib=all_features_unihill.iloc[0:37, 25]
ar_sr=all_features_unihill.iloc[37:122, 25 ]
ac_afib=all_features_unihill.iloc[0:37, 26]/100
ac_sr=all_features_unihill.iloc[37:122, 26 ]/100
data = (g1, g2)
colors = ("red", "green")
groups = ("Afib", "Sinus")
fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

#for data, color, group in zip(data, colors, groups):
#    x, y = data
#    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

#plt.title('Matplot scatter plot')
#plt.legend(loc=2)
#plt.show()

plt.scatter(ar_afib, ac_afib, color='r',label='AF')
plt.scatter(ar_sr, ac_sr, color='g',label='Sinus')
plt.xlabel('Autoregression')
plt.ylabel('regularity index')
plt.show()

###########
fig = plt.figure()
df = pd.DataFrame(dict(x=full_dataset_shuffled.iloc[:, 25], y=full_dataset_shuffled.iloc[:, 26]/100, label=full_dataset_shuffled['labels']))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.xlabel('Autoregression')
plt.ylabel('regularity index')
plt.title('MODE AF+unihill')
plt.show()

fig = plt.figure()
df = pd.DataFrame(dict(x=all_features_unihill.iloc[:, 25], y=all_features_unihill.iloc[:, 26]/100, label=Y_unihill))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.xlabel('Autoregression')
plt.ylabel('regularity index')
plt.title('Unihill + TYKS AF')
plt.show()


fig = plt.figure()
df = pd.DataFrame(dict(x=full_dataset_shuffled.iloc[:, 24], y=full_dataset_shuffled.iloc[:, 26]/100, label=y_all))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.xlabel('DFA')
plt.ylabel('regularity index')
plt.show()


fig = plt.figure()
df = pd.DataFrame(dict(x=full_dataset_shuffled.iloc[:, 9], y=full_dataset_shuffled.iloc[:, 26]/100, label=y_all))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.xlabel('HRV')
plt.ylabel('regularity index')
plt.show()


fig = plt.figure()
df = pd.DataFrame(dict(x=full_dataset_shuffled.iloc[:, 24], y=full_dataset_shuffled.iloc[:, 26]/100, label=y_all))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.xlabel('Entropy_Spectral')
plt.ylabel('regularity index')
plt.show()

##########################################
#
#y_pred_train = cross_val_predict(model_NN, X_all_norm,y_all , cv=loocv)
#plt.close('all')
#
#np.set_printoptions(precision=2)
#class_names=np.array(['Sinus','AFib'])
## Plot non-normalized confusion matrix
#plot_confusion_matrix(full_dataset_shuffled['labels'], y_pred_train, classes=class_names, title='Confusion matrix ML, without normalization')
#
## Plot normalized confusion matrix
#plot_confusion_matrix(full_dataset_shuffled['labels'], y_pred_train, classes=class_names, normalize=True,title='Confusion matrix ML,Normalized confusion matrix')
#plt.show()









end = time.time()
print(end - start)