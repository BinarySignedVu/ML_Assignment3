#--------------------------------------------------------------------------
# AUTHOR: Vu Nguyen
# FILENAME: svm.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: 9
#--------------------------------------------------------------------------*/

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

# Separate feature vectors and labels
X_training = np.array(df.values)[:, :64]  # features
y_training = np.array(df.values)[:, 64]   # labels

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the test data by using Pandas library

# Separate feature vectors and labels
X_test = np.array(df.values)[:, :64]  # features
y_test = np.array(df.values)[:, 64]   # labels

# Variables to hold the highest accuracy and best parameters
highest_accuracy = 0
best_params = {}

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for c_value in c:
    for degree_value in degree:
        for kernel_value in kernel:
            for decision_shape in decision_function_shape:
                # Create an SVM classifier with the current set of hyperparameters
                clf = svm.SVC(C=c_value, degree=degree_value, kernel=kernel_value, decision_function_shape=decision_shape)

                # Fit SVM to the training data
                clf.fit(X_training, y_training)

                # Predict the labels for the test set
                y_pred = clf.predict(X_test)

                # Calculate the accuracy of the predictions
                accuracy = np.sum(y_pred == y_test) / len(y_test)

                # Check if the calculated accuracy is higher than the previously one calculated
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_params = {'C': c_value, 'degree': degree_value, 'kernel': kernel_value, 'decision_function_shape': decision_shape}
                    print(f"Highest SVM accuracy so far: {highest_accuracy}, Parameters: C={c_value}, degree={degree_value}, kernel= {kernel_value}, decision_function_shape = '{decision_shape}'")
