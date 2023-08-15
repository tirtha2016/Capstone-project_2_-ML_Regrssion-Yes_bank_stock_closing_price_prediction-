# Capstone-project_2_-ML_Regrssion-Yes_bank_stock_closing_price_prediction-

![yes bank](https://github.com/tirtha2016/Capstone-project_2_-ML_Regrssion-Yes_bank_stock_closing_price_prediction-/assets/130783172/23dc0d2d-6d7a-413d-9ed2-9855e6d36c12)

# Project Summary -

Yes Bank is an Indian bank headquartered in Mumbai, India and was founded by Rana Kapoor and Ashok Kapoor in 2004.It is a well-known bank in Indian financial domain . It operates in Retail, MSME and Corporate banking sectors. It offers wide range of differentiated products for corporate and retail customers through retail banking and asset management services.

Sinces 2018, it has been in the news because of the fraud case involving Rana Kapoor. Owing to this fact, it was interesting to see how that impacted the stock prices of the company and whether Time series models or any other predictive models can do justice to such situations. This dataset has monthly stock prices of the bank since its inception and includes closing, starting, highest, and lowest stock prices of every month.

So "Machine Learning" is helping us to resolve the issue of all those companies and firms who want to gather some courage in order to survive in the market for a longer time. By predicting the price with the acquaintance of Machine Learning especially the linear Regression and other regressors, which helped firms and companies to sustain in the market. In this project the monthly Open,Close,Low and High prices of Yes Bank stock have helped to train the model on which learning occurred and then the respective prediction occurs.

This project is done by our team consist of two team members namely , RADHIKA DWIVEDI and TIRTHA BOSE

Here In this project we get a dataset contains five features like Date,Open,High,Close and in this we have to observe the dataset and comes to a conclusion to best suitable regression model which will provides us the best fit lines (best predicted values). At first we uploaded that data set in our colab and read the csv file given by almabetter team . Then we checked the shape and size of data and got to know that there are 185 rows and five columns(features) .Then we describe the statistical information about our data and here we got to know that our data is not normally distributed.Then we started Exploratory Data Analysis(EDA) to get some important insights from that.

Then we implement 3 regression model

Linear regression

lasso regression

Ridge regression

Also we have to use following evaluation metrics:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

R-squared

#  Problem Statement
As we are very much familiar with the situation of "YES BANK" that it has experienced significiant volatility and faced challenges , including a high profile fraud case involving Rana Kapoor.

So according to it our main objective of this project is to develop a reliable prediction model thta can forecast the closing price of YES BANK stock based on its historical data and relevant market indicators. Accurate prediction will help investors and traders make informed decisions , optimise their investment strategies and potentially maximize their returns.

Root Mean Squared Error (RMSE) 

Mean Absolute Percentage Error (MAPE)

Then Icalculated all the evaluation matrics for all the model and then got best suitable model and best predicted values.

# Main Libraries Used 

import pandas as pd

import numpy as np

from numpy import math

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from datetime import datetime

import missingno as msno

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import Lasso, LassoCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import *

![ml project](https://github.com/tirtha2016/Capstone-project_2_-ML_Regrssion-Yes_bank_stock_closing_price_prediction-/assets/130783172/f89e3f15-515f-4bf3-9a06-c80ac237adc9)


