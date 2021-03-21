#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
import io #used this is colab

#Files reading
dataset = pd.read_csv('music_train.csv')

#Data Preprocessing
titles = list(dataset.columns)
titles[-1],titles[-2] = titles[-2],titles[-1]
dataset = dataset[titles]
dataset = dataset.fillna(dataset.median())

#Differentiation in features and target
x = dataset.iloc[:, 1:26].values #dropping id from the features dataset
y = dataset.iloc[:, -1].values

#Validation dataset creation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 0)

#RandomForestClassifier Algorithm
from sklearn.ensemble import RandomForestClassifier
md = RandomForestClassifier(n_estimators = 1000) #increasing number of estimators to increase accuract
md.fit(x_train, y_train)
#higher the estimators, more time and resources the system uses

#Accuracy printing
print("Test Accuracy: ", md.score(x_test, y_test))
y_p = md.predict(x_test)
print("Predicted for validation data set:", y_p)


#Test set Predictions
testing = pd.read_csv('music_test.csv')
#Preprocessing
testing = testing.fillna(testing.median())
testing_x = testing.iloc[:, 1:26] #dropping id from the dataset

#Final Predictions
y_pred_fin = md.predict(testing_x)
print("Final test predictions: ", y_pred_fin)

#Creating CSV files
shrek = { 'id': testing['id'],
       'genre' : y_pred_fin}
predicted = pd.DataFrame(shrek)
predicted = predicted.astype(int)
print(predicted.head())
predicted.to_csv('shreya_rf6.csv', index = False) #Removing index, generating file
