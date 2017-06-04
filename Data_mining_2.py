# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:45:34 2017

@author: Kartikay89Singh
"""

# Data mining assignment 2 + Kaggle's Expedia competition

# Data preparation:

# Importing Libraries:
import numpy as np                          # Mathematical tool
import matplotlib.pyplot as plt             # to draw charts
import pandas as pd                         # for data processing and read csv

# Importing the dataset
dataset_train = pd.read_csv('training_set_VU_DM_2014.csv')
#dataset_test = pd.read_csv('test_set_VU_DM_2014.csv')
#dataset_train00 = pd.read_csv('training_set_VU_DM_2014-000.csv')

# columns information
print(dataset_train.info())
# These are the columns that were initially used to predict the results.
# columns to be used.
#date_time
#visitor_location_country_id
#prop_id
#prop_starrating
#prop_review_score
#prop_location_score1
#price_usd
#srch_destination_id
#srch_length_of_stay
#srch_booking_window
#srch_adults_count
#srch_children_count
#orig_destination_distance

#click_bool
#booking_bool
from pandas import DataFrame

dataset_new = dataset_train[['date_time', 'visitor_location_country_id', 'prop_id', 'prop_starrating' ,'prop_review_score', 'prop_location_score1', 'price_usd', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'orig_destination_distance', 'click_bool', 'booking_bool'] ]
print(dataset_new.head())

print(dataset_new.info())

#Distribution of ratings given by visitors
visitor_star = dataset_new['visitor_hist_starrating']
visitor_star = pd.to_numeric(visitor_star)

search_id = dataset_new['srch_id']
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
dataset_new_prop = dataset_new.sort('prop_id')
dataset_new_prop.head()

prop_id = dataset_new_prop.prop_id
prop_brand_bool = dataset_new_prop.prop_starrating
brand = []

print(len(brand))
print(len(prop_brand_bool))
chain = np.sum(brand)
print("Number of hotels of a chian: ", chain)

#Distribution of property ratings
plt.hist(dataset_new['prop_starrating'], 6, normed=True)
plt.ylabel('Density')
plt.xlabel('Property rating')
plt.savefig('prop_rating.png')

#number of hotels
hotels = dataset_new.prop_id
n_hotels = len(set(hotels))
print(n_hotels)

#number of countries
countries = dataset_new.prop_country_id
n_countries = len(set(countries))
print(n_countries)


plt.hist(dataset_new_prop.diff_rating.dropna(), normed=True)
plt.ylabel('Density')
plt.xlabel('Difference between visitor and property rating')
plt.savefig('dist_diff_rating')

fig, ax = plt.subplots()
plt.hist(dataset_new['month'], bins=range(1,14), align='left', normed=True)
plt.ylabel('Density')
plt.xlabel('Months')
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], size='small', ha='center')
plt.savefig('booking_month.png')

# Solving the date and time problem.
dataset_new['date'] = pd.to_datetime(dataset_new['date_time'],
format = "%Y-%m-%d %H:%M:%S")
dataset_new['month'] = dataset_new.date.dt.month

dataset_new = dataset_new[['visitor_location_country_id', 'prop_id', 'prop_starrating' ,'prop_review_score', 'prop_location_score1', 'price_usd', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'orig_destination_distance', 'month', 'click_bool', 'booking_bool'] ]

#creating objects to be used in the machine learning algorithm.

X = dataset_new.iloc[:, 0:13].values

y = dataset_new.iloc[:,13].values #+dataset_new.iloc[:,14].values

# Checking for null(NaN) values.
dataset_new.columns[dataset_new.isnull().any()]

# Handling the missing value:
    # prop.review_score.
mean01 = dataset_new.prop_review_score.mean()
dataset_new['prop_review_score'] = dataset_new.prop_review_score.fillna(mean01)
    # orig_destination_distance.
mean02 = dataset_new.orig_destination_distance.mean()
dataset_new['orig_destination_distance'] = dataset_new.orig_destination_distance.fillna(mean02)

  # Modelling: Regression

# Splitting the dataset into the training set and test set.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#X_train.reshape()
#y_train.reshape()
# Linear Regression.

# Regression model.- analysis.

# y = b0 + b1 * x1
#i,e.,
#Dependent variable = constant + coefficient * Independent variable.

# Fitting regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(X_test)
