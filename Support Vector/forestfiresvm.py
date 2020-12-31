#classify the Size_Categorie using SVM
#month	month of the year: 'jan' to 'dec'
#day	day of the week: 'mon' to 'sun'
#FFMC	FFMC index from the FWI system: 18.7 to 96.20
#DMC	DMC index from the FWI system: 1.1 to 291.3
#DC	DC index from the FWI system: 7.9 to 860.6
#ISI	ISI index from the FWI system: 0.0 to 56.10
#temp	temperature in Celsius degrees: 2.2 to 33.30
#RH	relative humidity in %: 15.0 to 100
#wind	wind speed in km/h: 0.40 to 9.40
#rain	outside rain in mm/m2 : 0.0 to 6.4
#Size_Categorie 	the burned area of the forest ( Small , Large)

#importing all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#for SVM algorithm importing libraries and functions
from sklearn.svm import SVC

#Loading dataset forestfires.csv and checking its information 
forest = pd.read_csv("forestfires.csv")
forest.head()

forest.info()

#target variable size_category information and counts
forest.size_category.describe()
forest.size_category.value_counts()

#EDA
forest.isnull().sum()

#Checking Dataset 
forest.head()

forest.month.value_counts()

forest.day.value_counts()

#Preprocessing Data by proving labels to categorical data
string_columns=["month","day"]
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    forest[i] = number.fit_transform(forest[i])
    forest[i] = number.fit_transform(forest[i])

forest.size_category.value_counts().plot(kind="bar")

#Splitting dataset as train and test
from sklearn.model_selection import train_test_split
train,test = train_test_split(forest,test_size = 0.3,random_state=30)
trainX = train.drop(["size_category"],axis=1)
trainY = train["size_category"]
testX = test.drop(["size_category"],axis=1)
testY = test["size_category"]

#Model Building
# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
# kernel = linear

model_linear = SVC(kernel = "linear")
model_linear.fit(trainX,trainY)
pred_test_linear = model_linear.predict(testX)

np.mean(pred_test_linear==testY) # Accuracy = 96.79

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(trainX,trainY)
pred_test_poly = model_poly.predict(testX)

np.mean(pred_test_poly==testY) # Accuracy = 0.75

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(trainX,trainY)
pred_test_rbf = model_rbf.predict(testX)

np.mean(pred_test_rbf==testY) # Accuracy = 0.737

#we select model build by Kernel=linear