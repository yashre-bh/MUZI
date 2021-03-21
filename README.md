# MUZI
1. Libraries used for this project are Pandas, MatplotLib, numpy, seaborn, and sklearn
2. The given files, music_train.csv and music_test.csv were imported/uploaded in the google colab/ pyCharm
3. So first of all, data preprocessing:
    a. Printed the data, analysed it for missing values and, noted the number of rows and columns
    b. Now, genre being the target attribute, should ideally be placed at the and of the data set. So I interchanged the       'topic' and 'genre' columns for more clarity
    c. The missing values in the given data set have been filled using the median method, that is, all the values of the        relevant columns were grouped, the median values was searched and filled up in the missing data columns
4. The features and the target attribute were seperated using iloc, I dropped the ID column from the feature data set, as    it was independent of the genre of the song (seen using the heat map made) (https://github.com/yashre-bh/MUZI/issues/1#issue-837112852)
5. Now from sklearn.model_selection, train_test_split was imported so as to facilitate easy creation of a validation data    set from the given data, i.e music.train.
6. The training set has attributes x_train and y_train
7. the testing validation set has attributes x_test and y_test (30% of the original dataset)
8. So, now, different models were used for training the machine, as follows:

    a. The KNN Model: K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning        technique. K-NN algorithm assumes the similarity between the new case/data and available cases and put the new            case into the category that is most similar to the available categories. K-NN algorithm stores all the available          data and classifies a new data point based on the similarity. This means when new data appears then it can be            easily classified into a well suite category by using K- NN algorithm.
       This algorithm resulted in an abysmal accuracy score of 0.23 on the training set
       
    b. The SVM Model: “Support Vector Machine” (SVM) is a supervised machine learning algorithm which can be used for      both classification or regression challenges. However,  it is mostly used in classification problems. In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiates the two classes very well.
    This model too resulted in a low accuracy of 0.228 on the training set.

    c. Multivariable Linear regression: Multiple linear regression refers to a statistical technique that is used to predict the outcome of a variable based on the value of two or more variables.
    This resulted in an accuracy of 0.07 on the given training set
    
    Plotting various graphs, I figured that relation between the attributes is not exactly, linear. Moreover, this was a classification task.
    
    Finally after trial and error, I arrived at the conclusion of using RandomForestClassifier, which gave a low yet acceptable accuracy score
    
    Random forest is a supervised learning algorithm. The "forest" it builds, is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result. It builds multiple decision trees and merges them together to get a more accurate and stable prediction.
    
    
9. Imported the necessary libraries
10. So, I tried increasing the n_estimators from 10 to 100 to 1000 to finally 10000, only to find that larger the number, slower the processing is.
11. Printed the accuracy for the training set, which comes out to be 0.43546872064625214 for n_estimators=1000. I did try submitting files in cerebro at numbers larger than that.
12. Trained the machine using x_train and y_train
13. Filled up the missing values in the test dataset, and printed the predicted values.
14. Created another dataset, and generated the csv file for the predictions.
15. Initially submitted 3 files without removing the index, resulting in a zero match score. 
16. Submitted the csv which approx gives an accuracy of 0.4 on cerebro.
