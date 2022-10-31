#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:37:12 2019

@author: Mojtaba Jafaritadi, PhD
"""


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


############################################### load dataset (including 422 subjects (AFib=187))


fs=200 #Hz


full_dataset_smartphone=load_dict_from_hdf5('dataset_afib_classification_422subjects.h5')

all_true_labels=np.concatenate((np.array(Y_modeaf),np.array(Y_unihill)))


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

loocv = model_selection.LeaveOneOut() 
X_all_norm=sc.fit_transform(full_dataset_shuffled.loc[:, 'mean_nni':'reg_index'])
y_pred_train = cross_val_predict(model_NN, X_all_norm, full_dataset_shuffled['labels'], cv=loocv)
plt.close('all')

np.set_printoptions(precision=2)
class_names=np.array(['Sinus','AFib'])
# Plot non-normalized confusion matrix
plot_confusion_matrix(full_dataset_shuffled['labels'], y_pred_train, classes=class_names, title='Confusion matrix ML, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(full_dataset_shuffled['labels'], y_pred_train, classes=class_names, normalize=True,title='Confusion matrix ML,Normalized confusion matrix')
plt.show()


########################################## Machine learning part


############################################ Training different classifiers for LOOCV
clf1 = model_AB.fit(X_train_norm, y_train)
clf2 = model_GB.fit(X_train_norm, y_train)
clf3 = model_RF.fit(X_train_norm, y_train)
clf4 = model_NN.fit(X_train_norm, y_train)
clf5 = GaussianNB().fit(X_train_norm, y_train)

eclf = VotingClassifier(estimators=[('ab', clf1), ('gb', clf2), ('rf', clf3), ('lr', clf4), ('gnb', clf5)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Adaptive Boosting', 'Gradient Boosting', 'Random Forest', 'Neural Network','Naive Bayes','Ensemble']):
    scores = cross_val_score(clf, X_all_norm, full_dataset_shuffled['labels'], cv=loocv, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
# 
    
    
 ############splitting the data into train and test (randomly)
dict_of_results = {}
for index in range(100): 
    tempDict = {}
    iterations = 'iteration' + str(index)
    clf1 = model_AB.fit(X_train_norm, y_train)
    clf2 = model_GB.fit(X_train_norm, y_train)
    clf3 = model_RF.fit(X_train_norm, y_train)
    clf4 = model_NN.fit(X_train_norm, y_train)
    clf5 = GaussianNB().fit(X_train_norm, y_train)
    X_train, X_test, y_train, y_test = train_test_split(full_dataset_shuffled.loc[:, 'mean_nni':'reg_index'], full_dataset_shuffled['labels'], test_size=0.3, random_state=np.random.randint(low=1, high=1000))
    
    sc = StandardScaler()
    X_train_norm = sc.fit_transform(X_train)
    X_test_norm = sc.transform (X_test)
       
    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Adaptive Boosting', 'Gradient Boosting', 'Random Forest', 'Neural Network','Naive Bayes','Ensemble']):
        model=clf.fit(X_train_norm, y_train)
        scores = model.score(X_test_norm, y_test)
#        values = [scores.mean(), scores.std()]
        tempDict[label] = scores.mean()    
        dict_of_results[iterations] = tempDict
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        
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
