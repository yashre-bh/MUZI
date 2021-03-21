import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
import io

dataset = pd.read_csv('music_train.csv')
titles = list(dataset.columns)
titles[-1],titles[-2] = titles[-2],titles[-1]
dataset = dataset[titles]

dataset = dataset.fillna(dataset.median())

x = dataset.iloc[:, 1:26].values #dropping id from the features dataset
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
md = RandomForestClassifier(n_estimators = 1000) #increasing number of estimators to increase accuract
md.fit(x_train, y_train)
#higher the estimators, more time and resources the system uses

print("Test Accuracy: ", md.score(x_test, y_test))
y_p = md.predict(x_test)
print("Predicted for validation data set:", y_p)

testing = pd.read_csv('music_test.csv')
testing = testing.fillna(testing.median())
testing_x = testing.iloc[:, 1:26] #dropping id from the dataset

y_pred_fin = md.predict(testing_x)
print("Final test predictions: ", y_pred_fin)

shrek = { 'id': testing['id'],
       'genre' : y_pred_fin}
predicted = pd.DataFrame(shrek)
predicted = predicted.astype(int)
print(predicted.head())
predicted.to_csv('shreya_rf6.csv', index = False)