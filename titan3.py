# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:16:58 2019

@author: mkyh8
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('titan_train.csv')
#dataset = dataset.dropna(subset = ['Age'])
dataset1 = dataset[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
dataset1 = pd.get_dummies(dataset1)

datatest= pd.read_csv('titan_test.csv')
#datatest = datatest.dropna(subset = ['Age', 'Fare'])
datatest1 = datatest[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
datatest1 = pd.get_dummies(datatest1)


X_train = dataset1.iloc[:,1:].values
Y_train = dataset1.iloc[:,0].values

X_test = datatest1.iloc[:,1:].values
Y_test = datatest1.iloc[:,0].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN', strategy = 'mean' , axis =0)
X_train[:,1:2] = imputer.fit_transform(X_train[:,1:2]) #fit_transform requires an array not a vector (!X_train[:,1] but !X_train[:,1])
X_test[:,1:3] = imputer.fit_transform(X_test[:,1:3])

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

conm = confusion_matrix(Y_test, Y_pred)
score = accuracy_score(Y_test, Y_pred)

data = {'PassengerId': datatest['PassengerId'], 'Survived': Y_pred}

submission = pd.DataFrame(data)

submission.to_csv('Submission.csv', index = False)