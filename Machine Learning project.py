# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#Importing dataset
dataset = pd.read_csv('ml1data.train.csv')
trainset = pd.read_csv('ml1data.test_unlabeled.csv',header=None)

X = pd.get_dummies(dataset.iloc[:, :-1])
y = dataset.iloc[:, 21].values
Z = pd.get_dummies(trainset.iloc[:, :])

#SVM 
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X= sc.fit_transform(X)
Z= sc.transform(Z)

from sklearn.svm import SVC
classifier= SVC(kernel ='linear',random_state =0)
classifier.fit(X,y)
y_presd = classifier.predict(Z)
y_pred = pd.DataFrame(y_presd)
CombineData = pd.DataFrame(np.append(trainset,y_pred,axis=1))


  
