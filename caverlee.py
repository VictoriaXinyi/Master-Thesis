#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:03:24 2019

@author: sunqiaoyubing
"""

## caverlee dataset ###

## import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Imports
import glob
import string
import ast

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
##%matplotlib inline

# statistical modelling
import statsmodels.api as sm

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


#metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report


## 1. import csv file
botornot1 = pd.read_csv('/Users/sunqiaoyubing/Desktop/botornot11.csv',sep=';')
train, test = train_test_split(botornot1, test_size=0.4, random_state=25)

#botornot1
train.shape, test.shape
train.info()

## 2. visualization
#1.
plt.title('Number of bot vs Number of Humans', y=1.1, size=15)
sns.countplot('Botornot', data=botornot1)

#2.
plt.title('Contrast of Followers Count for Bots and Humans', size=20, y=1.1)
df1 = botornot1[['followers_count','Botornot']]
bot_len1  = df1.ix[(df1['Botornot']==1)]
bot_len0  = df1.ix[(df1['Botornot']==0)]
p1=sns.kdeplot(bot_len0['followers_count'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['followers_count'], shade=True, color="b",label='Bots')

#3.
plt.title('Contrast of Followings Count for Bots and Humans', size=20, y=1.1)
df1 = botornot1[['followings_count','Botornot']]
bot_len1  = df1.ix[(df1['Botornot']==1)]
bot_len0  = df1.ix[(df1['Botornot']==0)]
p1=sns.kdeplot(bot_len0['followings_count'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['followings_count'], shade=True, color="b",label='Bots')

#4.
plt.title('Contrast of Tweets Count for Bots and Humans', size=20, y=1.1)
df1 = botornot1[['Tweets_count','Botornot']]
bot_len1  = df1.ix[(df1['Botornot']==1)]
bot_len0  = df1.ix[(df1['Botornot']==0)]
p1=sns.kdeplot(bot_len0['Tweets_count'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['Tweets_count'], shade=True, color="b",label='Bots')

#5.
plt.title('Contrast of screen name length for Bots and Humans', size=20, y=1.1)
df1 = botornot1[['screen_name_length','Botornot']]
bot_len1  = df1.ix[(df1['Botornot']==1)]
bot_len0  = df1.ix[(df1['Botornot']==0)]
p1=sns.kdeplot(bot_len0['screen_name_length'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['screen_name_length'], shade=True, color="b",label='Bots')

#7.
plt.title('Contrast of description length for Bots and Humans', size=20, y=1.1)
df1 = botornot1[['description_length','Botornot']]
bot_len1  = df1.ix[(df1['Botornot']==1)]
bot_len0  = df1.ix[(df1['Botornot']==0)]
p1=sns.kdeplot(bot_len0['description_length'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['description_length'], shade=True, color="b",label='Bots')

##
xset = train
X_all = xset.drop(['UserID','Botornot'], axis=1)
y_all = xset['Botornot']
X_all.shape
y_all.shape

#8.
f,ax = plt.subplots(figsize=(8, 5))
sns.heatmap(X_all.corr(), cmap='coolwarm', annot=True, linewidths=.5, fmt= '.1f',ax =ax)



X_1 = train.drop(['UserID'], axis=1)
sns.pairplot(X_1,vars=['followings_count','followers_count','Tweets_count','screen_name_length','description_length'],
             hue='Botornot',palette="husl",markers=["o", "+"]);
             
## 3. Modelling 
# features
features = ['followings_count','followers_count','Tweets_count','screen_name_length','description_length']
X_train = xset[features]
y_train = xset['Botornot']
X_test = test[features]
y_test = test['Botornot']
        

# standardize the features
fs_m1 = StandardScaler().fit(X_train)
Xs_train = fs_m1.transform(X_train)

# Model
logit_model=sm.Logit(y_train,Xs_train)
result=logit_model.fit()
print(result.summary2())

# prediction
logreg = LogisticRegression()
logreg.fit(Xs_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Compute precision, recall, F-measure and support
print(classification_report(y_test, y_pred))

# ROC curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()



















































