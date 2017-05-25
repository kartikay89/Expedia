# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:19:12 2017

@author: Kartikay89Singh
"""
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading the new randomly balanced dataset
random_bal = pd.read_csv('random_balanced_data.csv')

# Checking for null(NaN) values.
random_bal.columns[random_bal.isnull().any()]

# creating objects to be used in creating the model.

X2 = X = random_bal.iloc[:,0:21].values
                    

                    
y2 = random_bal.iloc[:,21].values 
                    

# Splitting the dataset into train and test set
from sklearn.cross_validation import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.25, random_state = -0)
                    

# Modelling

# Linear Regression

from sklearn.linear_model import LinearRegression

Regressor = LinearRegression()
Regressor.fit(X2_train, y2_train)

# predicting the result for the test set
y_pred2 = Regressor.predict(X2_test)


# 5-fold cross validation
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

seed = 5

kfold = model_selection.KFold(n_splits=5, random_state=seed)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X2_train, y2_train, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


















