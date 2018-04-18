# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 20:15:41 2018

@author: 13603
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:43:29 2017

@author: kanwei
"""
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn import svm
from sklearn import neighbors 
from sklearn import tree
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import GradientBoostingClassifier
#import matplotlib.pyplot as plt



load_x = sio.loadmat('ori_liking.mat')
load_y = sio.loadmat('y.mat')
x = load_x['ori_liking']
y = load_y['y']
y = y[:,3]

#load_x = sio.loadmat('ori_arousal.mat')
#load_y = sio.loadmat('y.mat')
#x = load_x['ori_arousal']
#y = load_y['y']
#y = y[:,1]

# test on each subject
accuracy1 = []
accuracy2 = []
accuracy3 = []
accuracy4 = []
accuracy5 = []
accuracy6 = []


#scaler = preprocessing.MinMaxScaler().fit(x)
#x = scaler.transform(x)
for i in range(32):
    x_test = x[40*i:40*(i+1),:] 
    y_test = y[40*i:40*(i+1)]
    if i ==0:
        x_train = x[40:,:]
        y_train = y[40:]
    elif i==32:
        x_train = x[:40*31,:]
        y_train = y[:40*31]
    else: 
        x_train = np.vstack((x[0:40*i,:],x[40*(i+1):,:]))
        y_train = np.hstack((y[0:40*i],y[40*(i+1):]))
    test_data  = (x_test,y_test)
    train_data = (x_train,y_train)
   
    #SVM
#    clf1 = svm.SVC(kernel = 'rbf')
#    clf1.fit(x_train,y_train)
#    pre_y1 = clf1.predict(x_test)
#    acc1 = float(sum(pre_y1==y_test)/40.)
#    accuracy1.append(acc1)
##print('SVM ave_accuracy',  np.mean(accuracy,dtype=np.float16))
##print('std',np.std(accuracy,dtype=np.float16))
#    
#    #KNN
#    clf2 = neighbors.KNeighborsClassifier(n_neighbors=5)
#    clf2.fit(x_train, y_train) 
#    pre_y2 = clf2.predict(x_test)  
#    acc2 = float(sum(pre_y2==y_test)/40.)
#    accuracy2.append(acc2)
##print('KNN ave_accuracy',  np.mean(accuracy,dtype=np.float16))
##print('std',np.std(accuracy,dtype=np.float16))
#    
     #decision tree
    clf3 = tree.DecisionTreeClassifier()
    clf3.fit(x_train,y_train )
    pre_y3 = clf3.predict(x_test)
    acc3 = float(sum(pre_y3==y_test)/40.)
    accuracy3.append(acc3)
#print(' DTR ave_accuracy',  np.mean(accuracy,dtype=np.float16))
#print('std',np.std(accuracy,dtype=np.float16))

    #logistic regression
    clf4 = LogisticRegression(penalty = 'l2',solver = 'liblinear',) 
    clf4.fit(x_train, y_train)  
    pre_y4 = clf4.predict(x_test)
    acc4 = float(sum(pre_y4==y_test)/40.)
    accuracy4.append(acc4)

    #naive_bayes
    clf5 = GaussianNB()
    clf5.fit(x_train, y_train)
    pre_y5 = clf5.predict(x_test)
    acc5 = float(sum(pre_y5==y_test)/40.)
    accuracy5.append(acc5)
#print('NB ave_accuracy',  np.mean(accuracy,dtype=np.float16))
#print('std',np.std(accuracy,dtype=np.float16))
 
#     #GBDT
    clf6 = GradientBoostingClassifier()
    clf6.fit(x_train, y_train)
    pre_y6 = clf6.predict(x_test)
    acc6 = float(sum(pre_y6==y_test)/40.)
    accuracy6.append(acc6)

#
#print('SVM ave_accuracy',  np.mean(accuracy1,dtype=np.float16))
#print('std',np.std(accuracy1,dtype=np.float16))
#
#print('KNN ave_accuracy',  np.mean(accuracy2,dtype=np.float16))
#print('std',np.std(accuracy2,dtype=np.float16))


#
print(' DTR ave_accuracy',  np.mean(accuracy3,dtype=np.float16))
print('std',np.std(accuracy3,dtype=np.float16))

print('LR ave_accuracy',  np.mean(accuracy4,dtype=np.float16))
print('std',np.std(accuracy4,dtype=np.float16))

print('NB ave_accuracy',  np.mean(accuracy5,dtype=np.float16))
print('std',np.std(accuracy5,dtype=np.float16))

print('GBDT ave_accuracy',  np.mean(accuracy6,dtype=np.float16))
print('std',np.std(accuracy6,dtype=np.float16))
