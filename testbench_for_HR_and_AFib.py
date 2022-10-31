#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:48:35 2019

@author: Mojtaba Jafaritadi

"""

from HeartBeatDetector import *
from HRV_parameters import *
from complexity_estimators import *
from autoregressive_model import *
from afib_detector import *
from h5_helperfunctions import *

import pandas as pd

MODE_AF_signals=load_dict_from_hdf5('MODE_AF_rawsignals.h5')
UniHill_signals=load_dict_from_hdf5('UniHill_rawsignals.h5')

y_train=pd.read_hdf('labels_modeaf.h5','y_train')
y_test=pd.read_hdf('labels_unihillaf.h5','y_test')
Y_test=np.array(y_test['y_test'])
Y_train=np.array(y_train['y_train'])

list_of_key_MODEAFnames=list(MODE_AF_signals.keys())     
list_of_key_UniHillnames=list(UniHill_signals.keys())     


############################################### training data (Mode-AF)
signals_fused=dict()
heart_beat_locs=dict()
HRV_features=dict()
AFib_autocorr=dict()
Complexity_features=dict()
AR_features=dict()

fs=200 #Hz

for i in list_of_key_MODEAFnames[:]:

    
    plt.close('all')
    
####################################### Processing Heartrates and getting features
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


HRV_feat_train=pd.DataFrame.from_dict(HRV_features, orient='index') ## transform HRV features to list
ACR_feat_train=pd.DataFrame.from_dict(AFib_autocorr, orient='index')# transform Autocorrelatio features to list
CPLX_feat_train=pd.DataFrame.from_dict(Complexity_features, orient='index')# transform Autocorrelatio features to list
AR_feature_train=pd.DataFrame.from_dict(AR_features, orient='index')# transform autoregressive features to list
all_features_train=pd.concat([HRV_feat_train,CPLX_feat_train,AR_feature_train],axis=1)


all_features_train['reg_index'] = ACR_feat_train['regularity_index'] # add  regularity index feature to HRV features



################################# TESTING DATA (UniHill+TYKS AF)

signals_fused=dict()
heart_beat_locs=dict()
HRV_features=dict()
AFib_autocorr=dict()
Complexity_features=dict()
AR_features=dict()

fs=200 #Hz

for i in list_of_key_UniHillnames[:]:

    
    plt.close('all')
    
####################################### Processing Heartrates and getting features
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


HRV_feat_test=pd.DataFrame.from_dict(HRV_features, orient='index') ## transform HRV features to list
ACR_feat_test=pd.DataFrame.from_dict(AFib_autocorr, orient='index')# transform Autocorrelatio features to list
CPLX_feat_test=pd.DataFrame.from_dict(Complexity_features, orient='index')# transform Autocorrelatio features to list
AR_feature_test=pd.DataFrame.from_dict(AR_features, orient='index')# transform autoregressive features to list
all_features_test=pd.concat([HRV_feat_test,CPLX_feat_test,AR_feature_test],axis=1)


all_features_test['reg_index'] = ACR_feat_test['regularity_index'] # add  regularity index feature to HRV features


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
import pandas as pd


#############################################################Preprocessing
sc = StandardScaler()

#all_features_test.reset_index(inplace = True)
#features_test=all_features_test.drop(columns='index')

X_train = sc.fit_transform(all_features_train)
X_test = sc.transform (all_features_test)

##################################################Cross Validation
loocv = model_selection.LeaveOneOut() 
model_RF = RandomForestClassifier(n_estimators=250,random_state=1368) #define random forest as a fitting model to classify AFib
model_LR= LogisticRegression() #define logistic regression  as the fitting model to classify AFib
model_GB = GradientBoostingClassifier(n_estimators=30, learning_rate=1.0, max_depth=1, random_state=0)
model_AB = AdaBoostClassifier(n_estimators=35)
model_GB = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)

y_pred_train = cross_val_predict(model_RF, X_train, Y_train, cv=loocv)

plt.close('all')

np.set_printoptions(precision=2)
class_names=np.array(['Sinus','AFib'])
# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_train, y_pred_train, classes=class_names, title='Confusion matrix ML, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_train, y_pred_train, classes=class_names, normalize=True,title='Confusion matrix ML,Normalized confusion matrix')
plt.show()
########################################### Feature importances
model_RF.fit(X_train, Y_train)
important_Features=model_AB.feature_importances_
data_top = list(all_features_train.columns)
data = {'Features':data_top, 'Weight':important_Features}
feature_importances = pd.DataFrame(data, columns = ['Features', 'Weight'])

importances = model_AB.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_AB.estimators_],
             axis=0)
indices = np.argsort(importances)

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.barh(range(all_features_train.shape[1]), importances[indices],
       color="r")
# If you want to define your own labels,
# change indices to a list of labels on the following line.
plt.yticks(range(all_features_train.shape[1]), feature_importances['Features'])
plt.show()

########################################## Testing the model
model_NN =MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(15,),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=100, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5,  random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)


clf1 = model_AB.fit(X_train, Y_train)
clf2 = model_GB.fit(X_train, Y_train)
clf3 = model_RF.fit(X_train, Y_train)
clf4 = model_NN.fit(X_train, Y_train)
clf5 = GaussianNB().fit(X_train, Y_train)


eclf = VotingClassifier(estimators=[('ab', clf1), ('gb', clf2), ('rf', clf3), ('lr', clf4), ('gnb', clf5)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Adaptive Boosting', 'Gradient Boosting', 'Random Forest', 'Neural Network','Naive Bayes','Ensemble']):
    scores = cross_val_score(clf, X_train, Y_train, cv=loocv, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
 
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Adaptive Boosting', 'Gradient Boosting', 'Random Forest', 'Neural Network','Naive Bayes','Ensemble']):
    model=clf.fit(X_test, Y_test)
    scores = model.score(X_train, Y_train)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
     
#for clf, label in zip([clf1, clf2, clf3, eclf], ['Adaptive Boosting', 'Gradient Boosting', 'Random Forest', 'Ensemble']):
#   
#    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


#########################
from matplotlib import pyplot
df = pd.DataFrame(dict(x=all_features_train[:,25], y=all_features_train[:,26], label=Y_train))
colors = {0:'sinus', 1:'Afib'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()