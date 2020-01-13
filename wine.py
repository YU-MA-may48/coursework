#!/usr/bin/env python
# coding:utf-8

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.stas import spearmanr
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

#import wine dataset from local
path= '/Users/mayu/OneDrive - Cardiff University/applied machine learning/Coursework/datasets_coursework1/Wine/wine_train.csv'
df=pd.read_csv(path,sep='\t')
df.head()

# x = np.array([7.2, 0.23, 0.32, 8.5, 0.058, 47, 186, 0.9956, 3.19, 0.4, 9.9]).reshape((-1, 1))
# y = np.array([1, 2, 3, 4, 5, 6,7,8,9])

# print (x)
# print (y)

# features for training
x_train=df[df.columns[0:11]]
y_train=df['quality']

regressor=linearRegression()
regressor.fit(x_train,y_train)

path_test='/Users/mayu/OneDrive - Cardiff University/applied machine learning/Coursework/datasets_coursework1/Wine/wine_test.csv'
#import test dataset
df_test = pd.read_csv(path_test,sep='\t')

#extract feature used for prediction
x_test=df_test[df_test.columns[0:11]]

#get the predicted value from the training model
y_pred=regressor.predict(x_test)

#get the actual "quality" value
y_test=df_test['quality']

#compare between predicted value and actual value
df_Result=pd.DataFrame({'Predicted Value': y_pred,'Actual Value': y_test})
df_Result.head()

# model = LinearRegression()
# model.fit(x, y)
# model = LinearRegression().fit(x, y)
# r_sq = model.score(x, y)
# print('coefficient of determination:', r_sq)
# print('intercept:', model.intercept_)
# print('slope:', model.coef_)
#
# new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
# print('intercept:', new_model.intercept_)
# print('slope:', new_model.coef_)
#
# y_pred = model.predict(x)
# #print('predicted response:', y_pred, sep= ' ')
# print("predicted response: \n" , y_pred)
#
# y_pred = model.intercept_ + model.coef_ * x
# #print('predicted response:', y_pred, sep='\n')
# print("predicted response: \n" , y_pred)


# x_new = np.arange(5).reshape((-1, 1))
# print(x_new)
#
# y_new = model.predict(x_new)
# print(y_new)

poly_reg=PolynomialFeatures(degree=2)

#transfer training and test features to polynomial format
x_poly=poly_reg.fit_transform(x_train)
x_poly_test=poly_reg.transform(x_test)

model=LinearRegression()
model.fit(x_poly,y_train)

y_poly_pred=model.predict(x_poly_test)
y_test=df_test['quality']

df_Result=pd.DataFrame({'Predicted Value':y_poly_pred, 'Actual Value': y_test })
df_Result.head()

print('----------------Linear Regression---------------')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.squr(metrics.mean_squared_error(y_test,y_pred)))
print('------------------------------------------------')

print('----------------Polynomial Regression---------------')
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.squr(metrics.mean_squared_error(y_test,y_pred)))
print('------------------------------------------------')
