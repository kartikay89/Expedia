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
dataset_new11 = dataset_train[['date_time', 'visitor_location_country_id', 'prop_id', 'prop_starrating' ,'prop_review_score', 'prop_location_score1', 'price_usd', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'orig_destination_distance', 'click_bool', 'booking_bool'] ]
print(dataset_new11.head())

print(dataset_new11.info())

#Distribution of ratings given by visitors
visitor_star = dataset_new1['visitor_hist_starrating']
visitor_star = pd.to_numeric(visitor_star)

search_id = dataset_new1['srch_id']
search_id = pd.to_numeric(search_id)

visitor_rating = []
for i in search_id:
    if search_id[i+1] != search_id[i]:
        x = visitor_star[i]
        visitor_rating.append(x)


visitor_rating = pd.Series(visitor_rating)
plt.hist(visitor_rating.dropna(), 6, normed=True)
plt.xlim(0,6)
plt.ylabel('Density')
plt.xlabel('Visitor rating')
plt.savefig('dist_vis_rating.png')

#sort on property ID
dataset_new1_prop = dataset_new1.sort('prop_id')
dataset_new1_prop.head()

prop_id = dataset_new1_prop.prop_id
prop_brand_bool = dataset_new1_prop.prop_starrating
brand = []

print(len(brand))
print(len(prop_brand_bool))
chain = np.sum(brand)
print("Number of hotels of a chian: ", chain)

#Distribution of property ratings
plt.hist(dataset_new1['prop_starrating'], 6, normed=True)
plt.ylabel('Density')
plt.xlabel('Property rating')
plt.savefig('prop_rating.png')

#number of hotels
hotels = dataset_new1.prop_id
n_hotels = len(set(hotels))
print(n_hotels)

#number of countries
countries = dataset_new1.prop_country_id
n_countries = len(set(countries))
print(n_countries)


plt.hist(dataset_new1_prop.diff_rating.dropna(), normed=True)
plt.ylabel('Density')
plt.xlabel('Difference between visitor and property rating')
plt.savefig('dist_diff_rating')

fig, ax = plt.subplots()
plt.hist(dataset_new1['month'], bins=range(1,14), align='left', normed=True)
plt.ylabel('Density')
plt.xlabel('Months')
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], size='small', ha='center')
plt.savefig('booking_month.png')


# Solving the date and time problem.
dataset_new11['date'] = pd.to_datetime(dataset_new11['date_time'],
format = "%Y-%m-%d %H:%M:%S")
dataset_new11['month'] = dataset_new11.date.dt.month

dataset1 = dataset_new11[['visitor_location_country_id', 'prop_id', 'prop_starrating' ,'prop_review_score', 'prop_location_score1', 'price_usd', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'orig_destination_distance', 'month', 'click_bool', 'booking_bool'] ]

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
