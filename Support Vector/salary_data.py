#Prepare a classification model using SVM for salary data 
#Data Description:
#age -- age of a person
#workclass	-- A work class is a grouping of work 
#education	-- Education of an individuals	
#maritalstatus -- Marital status of an individulas	
#occupation	 -- occupation of an individuals
#relationship -- 	
#race --  Race of an Individual
#sex --  Gender of an Individual
#capitalgain --  profit received from the sale of an investment	
#capitalloss	-- A decrease in the value of a capital asset
#hoursperweek -- number of hours work per week	
#native -- Native of an individual
#Salary -- salary of an individual

#importing all necessary libraries
import pandas as pd 
import numpy as np

#for SVM algorithm importing libraries and functions
from sklearn.svm import SVC

#Loading Dataset boh training and test
salary_train = pd.read_csv("SalaryData_Train.csv")
salary_test = pd.read_csv("SalaryData_Test.csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

#EDA
salary_train.isnull().sum()

salary_test.isnull().sum()

#Preprocessing Data by proving labels to categorical data
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]

#target variable size_category information and counts
salary_test.Salary.describe()
salary_test.Salary.value_counts()

trainX = salary_train.drop(["Salary"],axis=1)
trainY = salary_train["Salary"]
testX = salary_test.drop(["Salary"],axis=1)
testY = salary_test["Salary"]

from sklearn.preprocessing import scale
trainX= scale(trainX)
testX=scale(testX)

#Model Building
# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
# kernel = linear
model_poly = SVC(kernel = "poly")
model_poly.fit(trainX,trainY)
pred_test_poly = model_poly.predict(testX)

np.mean(pred_test_linear==testY) # Accuracy = 77.95

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(trainX,trainY)
pred_test_rbf = model_rbf.predict(testX)

np.mean(pred_test_rbf==testY) # Accuracy = 84.65

# kernel = sigmoid
model_rbf = SVC(kernel = "sigmoid")
model_rbf.fit(trainX,trainY)
pred_test_sigmoid = model_rbf.predict(testX)

np.mean(pred_test_sigmoid==testY) # Accuracy = 75.28

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X) #accuracy=0.81

#we select rbf model as final model as it has high accuracy #84.65
#Scaling dataset also increased accuracy from 77.55 to 84.65 by using rbf model
#linear model also worked using scale data but accuracy is less 0.81