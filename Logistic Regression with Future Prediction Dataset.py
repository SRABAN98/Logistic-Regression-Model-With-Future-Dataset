#import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import the dataset
dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\28th,29th\2.LOGISTIC REGRESSION CODE\Social_Network_Ads.csv")


#splitting the dataset into I.V and D.V
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)


#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)


#Predicting the Test set results
y_pred = classifier.predict(x_test)


#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac)


#This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr


#bias calculation
bias = classifier.score(x_train, y_train)
bias


#variance calculation
variance = classifier.score(x_test, y_test)
variance


#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#


#------------------------FUTURE PREDICTION--------------------------


#import the future prediction dataset
dataset1 = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\3.Aug\3rd\Future prediction1.csv")


#copy the future prediction dataset in to a new variable
d2 = dataset1.copy()


#clean the future prediction dataset for the further operation
dataset1 = dataset1.iloc[:,[2, 3]].values


#Feature Scalling of the future prediction dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
M = sc.fit_transform(dataset1)


#creating the future prediction dataframe
y_pred_LogisticRegression = pd.DataFrame()

d2["y_pred_LogisticRegression"] = classifier.predict(M)


#save the future prediction dataframe as the .csv file format
d2.to_csv("FPofLogitAlgo.csv")


#To get the path where exactly the predicted .csv file saved in our desktop
import os
os.getcwd()
