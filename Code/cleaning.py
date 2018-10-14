#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load dataset
train_y = pd.read_csv('../Dataset/y_train_sat6.csv', header=None)
train_x = pd.read_csv('../Dataset/X_train_sat6.csv', header=None)
test_y = pd.read_csv('../Dataset/y_test_sat6.csv', header=None)
test_x = pd.read_csv('../Dataset/X_test_sat6.csv', header=None)

#Checking df shape
# print('train_y shape',train_y.shape)
# print('train_x shape',train_x.shape)
# print('test_y shape',test_y.shape)
# print('test_x shape',test_x.shape)

#Checking for null values
# print('train_y has null',train_y.isnull().any().any())
# print('train_x has null',train_x.isnull().any().any())
# print('test_y has null',test_y.isnull().any().any())
# print('test_x has null',test_x.isnull().any().any())

#Checking largest and smallest values. For x should be between 0 and 255. for y should be between 0 and 1
# print('train_y range',train_y.describe())
# print('train_x range',train_x.describe())
# print('test_y range',test_y.describe())
# print('test_x range',test_x.describe())

#The data looks good with no null values and all values are within range. Reshapping for dataframe is needed for x sets
#to represent the original size of the images: 28x28 pixel with 4 color bands

#Data manipulation
#Reshapping data
train_x = train_x.values.reshape(-1,28,28,4)
test_x = test_x.values.reshape(-1,28,28,4)
print('train_x shape',train_x.shape)
print('test_x shape',test_x.shape)
