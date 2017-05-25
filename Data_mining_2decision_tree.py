# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:22:42 2017

@author: Kartikay89Singh
"""

# Decision tree
# Importing the libraries.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset.

dataset_train = pd.read_csv('training_set_VU_DM_2014.csv')

# creating the new dataset.
dataset_new1 = dataset_train[['date_time', 'visitor_location_country_id', 'prop_id', 'prop_starrating' ,'prop_review_score', 'prop_location_score1', 'price_usd', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'orig_destination_distance', 'click_bool', 'booking_bool'] ]
print(dataset_new1.head())

print(dataset_new1.info())

# Solving the date and time problem.
dataset_new1['date'] = pd.to_datetime(dataset_new1['date_time'],
format = "%Y-%m-%d %H:%M:%S")
dataset_new1['month'] = dataset_new1.date.dt.month

dataset1 = dataset_new1[['visitor_location_country_id', 'prop_id', 'prop_starrating' ,'prop_review_score', 'prop_location_score1', 'price_usd', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'orig_destination_distance', 'month', 'click_bool', 'booking_bool'] ]

# Adding new variables
dataset1['mean_price'] = dataset_train['price_usd'].groupby(dataset1['prop_id']).transform('mean')

dataset1['prop_price_diff'] = np.exp(dataset_train['prop_log_historical_price']) - dataset_train['price_usd']

dataset1['vist_price_diff'] = dataset_train['visitor_hist_adr_usd'] - dataset_train['price_usd']

dataset1['fee_person'] = (dataset_train['srch_room_count']*dataset_train['price_usd'])/(dataset_train['srch_adults_count']+dataset_train['srch_children_count'])

dataset1['total_fee'] = dataset_train['srch_room_count'] * dataset_train['price_usd']

#dataset2 = dataset1[['visitor_location_country_id', 'prop_id', 'prop_starrating' ,'prop_review_score', 'prop_location_score1', 'price_usd', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'orig_destination_distance', 'month', 'mean_price','prop_price_diff', 'vist_price_diff', 'fee_person', 'total_fee', 'prop_location_score2', 'promotion_flag','click_bool', 'booking_bool'] ]

# Checking for null(NaN) values.
dataset1.columns[dataset1.isnull().any()]


# Handling the missing value:
    # prop.review_score.
mean01 = dataset1.prop_review_score.mean()
dataset1['prop_review_score'] = dataset1.prop_review_score.fillna(mean01)
    # orig_destination_distance.
mean02 = dataset1.orig_destination_distance.mean()
dataset1['orig_destination_distance'] = dataset1.orig_destination_distance.fillna(mean02)
    # visit_price_diff:
mean03 = dataset1.vist_price_diff.mean()
dataset1['vist_price_diff'] = dataset1.vist_price_diff.fillna(mean01)

# creating objects to be used in creating the model.

X1 = X = dataset1.iloc[:,0:13].values
                    
y1 = dataset1.iloc[:,13].values


# Splitting the dataset into the training set and test set.
from sklearn.cross_validation import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

# creating the decision tree model from the dataset.
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X1_train, y1_train)

# predicting the results:

y_pred1 = regressor.predict(X1_test)


# k-fold cross validation.
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import  cross_val_score

dataset1 = DecisionTreeClassifier(min_samples_split = 20, random_state = 99)
dataset1.fit(X1_train, y1_train)
scores = cross_val_score(dataset1, X1_train, y1_train, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
scores.std()))
