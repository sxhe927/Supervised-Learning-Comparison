# This is for CS 7641 Machine Learning Assignment 1
# Sixin He (she89) @ Georgia Tech
# The following code is meant to do supervised learing for the Adult dataset
# Be sure all the libraries and packets are installed
# Five learning algorithms are implemented one by one, it is recommeded to run each of them and comment the others
#


import pydot
from sklearn import tree, neighbors, ensemble, cross_validation
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from StringIO import StringIO
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.validation import CrossValidator
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pprint import pprint
import warnings
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.learning_curve import learning_curve
warnings.filterwarnings("ignore", module="matplotlib")


import matplotlib.pyplot as plt


#training data is contained in "adult.data"

"""
the csv data is stored as such:
age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income
these values are explained in the file adult.names
our first step is to parse the data
"""

#first we define a set of conversion functions from strings to integer values because working with strings is dumb
#especially since the computer doens't care when doing machine learning
def create_mapper(l):
    return {l[n] : n for n in xrange(len(l))}

workclass = create_mapper(["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
education = create_mapper(["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
marriage = create_mapper(["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
occupation = create_mapper(["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
relationship = create_mapper(["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = create_mapper(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
sex = create_mapper(["Female", "Male"])
country = create_mapper(["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])
income = create_mapper(["<=50K", ">50K","<=50K.", ">50K."])

converters = {
    1: lambda x: workclass[x],
    3: lambda x: education[x],
    5: lambda x: marriage[x],
    6: lambda x: occupation[x],
    7: lambda x: relationship[x],
    8: lambda x: race[x],
    9: lambda x: sex[x],
    13: lambda x: country[x],
    14: lambda x: income[x]
}

"""
load the data into numpy
this section is also written for use in a
"""
train = "adult.data"
test = "adult.test"

def load(filename):
    with open(filename) as data:
        adults = [line for line in data if "?" not in line]  # remove lines with unknown data

    return np.loadtxt(adults,
                      delimiter=', ',
                      converters=converters,
                      dtype='u4',
                      skiprows=1
                      )


def start_adult():
    """
    tx - training x axes
    ty - training y axis
    rx - result (testing) x axes
    ry - result (testing) y axis
    """
    tr = load(train)
    te = load(test)
    X_train, y_train = np.hsplit(tr, [14])
    X_test, y_test = np.hsplit(te, [14])
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    for i in xrange(len(y_test)):
        y_test[i] -= 2
    # print np.unique(y_train)
    # print np.unique(y_test)
    j = 1
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = start_adult()

plt.hist(y_test, range=(0,1))
plt.hist(y_train, range=(0,1))
plt.xlabel('Quality')
plt.xticks([0, 1])
plt.ylabel('Count')
plt.title('Distribution of Adult Classes')
plt.show()

# # # decision trees
# training_accuracy = []
# validation_accuracy = []
# test_accuracy = []
# scoring = {'acc': 'accuracy',
#            'prec_macro': 'precision_macro',
#            'rec_micro': 'recall_macro'}
# for i in xrange(2,20):
#     classifier = DecisionTreeClassifier(max_depth=i, min_samples_split = 5)
#     classifier.fit(X_train, y_train)
#     # print classifier
#     scores = cross_validate(classifier, X_train, y_train, cv=10, scoring=scoring, return_train_score=True)
#     validation_accuracy.append(np.average(scores['train_acc']))
#     test_accuracy.append(np.average(scores['test_acc']))
#
# Kvals = range(2, 20)
# plt.figure()
# plt.title('Adult Decision Tree: Accuracy vs. max_depth')
# plt.plot(Kvals, validation_accuracy, '-', label='train_acc')
# plt.plot(Kvals, test_accuracy, 'x-', label='validation_acc')
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('max_depth')
# plt.ylabel('Accuracy')
# plt.show()

# estimator = DecisionTreeClassifier(max_depth=9, min_samples_split = 5)
#
# train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X_train, y_train,
#                    train_sizes=np.linspace(0.1, 1.0, 10),
#                    cv=10, scoring=None,
#                    exploit_incremental_learning=False,
#                    n_jobs=1, pre_dispatch="all", verbose=0)
# train_scores =np.average(train_scores,axis=1)
# test_scores = np.average(test_scores, axis=1)
#
# # i = 1
# plt.figure()
# plt.title('Adult Decision Tree: Accuracy vs. training_size')
# plt.plot(train_sizes_abs, train_scores, '-', label='Training score')
# plt.plot(train_sizes_abs, test_scores, 'x-', label='Cross-Validation score')
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('training_size')
# plt.ylabel('Accuracy')
# plt.show()
#
#
#
# estimator = DecisionTreeClassifier(max_depth=9, min_samples_split = 5)
# estimator.fit(X_train, y_train)
# y_pred = estimator.predict(X_test)
# acc_score = accuracy_score(y_test, y_pred, normalize=True)
# print acc_score

# KNN
# training_accuracy = []
# validation_accuracy = []
# test_accuracy = []
# scoring = {'acc': 'accuracy',
#            'prec_macro': 'precision_macro',
#            'rec_micro': 'recall_macro'}
#
#
# for k in xrange(1,30):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#     scores = cross_validate(knn, X_train, y_train, cv=10, scoring=scoring, return_train_score=True)
#     validation_accuracy.append(np.average(scores['train_acc']))
#     test_accuracy.append(np.average(scores['test_acc']))
#
#
# Kvals = range(1, 30)
# plt.figure()
# plt.title('Adult KNN: Accuracy vs. n_neighbors')
# plt.plot(Kvals, validation_accuracy, '-', label='train_acc')
# plt.plot(Kvals, test_accuracy, 'x-', label='validation_acc')
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('n_neighbors')
# plt.ylabel('Accuracy')
# plt.show()



#
# estimator = KNeighborsClassifier(n_neighbors=10)
#
# train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X_train, y_train,
#                    train_sizes=np.linspace(0.1, 1.0, 10),
#                    cv=10, scoring=None,
#                    exploit_incremental_learning=False,
#                    n_jobs=1, pre_dispatch="all", verbose=0)
# train_scores =np.average(train_scores,axis=1)
# test_scores = np.average(test_scores, axis=1)
#
# # i = 1
# plt.figure()
# plt.title('Adult KNN: Accuracy vs. training_size')
# plt.plot(train_sizes_abs, train_scores, '-', label='Training score')
# plt.plot(train_sizes_abs, test_scores, 'x-', label='Cross-Validation score')
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('training_size')
# plt.ylabel('Accuracy')
# plt.show()
#
#
#
# estimator = KNeighborsClassifier(n_neighbors=10)
# estimator.fit(X_train, y_train)
# y_pred = estimator.predict(X_test)
# acc_score = accuracy_score(y_test, y_pred, normalize=True)
# print acc_score
# #

# boosting
# training_accuracy = []
# validation_accuracy = []
# test_accuracy = []
# scoring = {'acc': 'accuracy',
#            'prec_macro': 'precision_macro',
#            'rec_micro': 'recall_macro'}
#
# for j in xrange(1,40):
#     AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9, min_samples_split=5), n_estimators=j, learning_rate = 1)
#     AdaBoost.fit(X_train, y_train)
#     scores = cross_validate(AdaBoost, X_train, y_train, cv=10, scoring=scoring, return_train_score=True)
#     validation_accuracy.append(np.average(scores['train_acc']))
#     test_accuracy.append(np.average(scores['test_acc']))
#
#
# Kvals = range(1, 40)
# plt.figure()
# plt.title('Adult Boosting: Accuracy vs. n_estimators')
# plt.plot(Kvals, validation_accuracy, '-', label='train_acc')
# plt.plot(Kvals, test_accuracy, 'x-', label='validation_acc')
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('n_estimator')
# plt.ylabel('Accuracy')
# plt.show()




# estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9, min_samples_split = 5), n_estimators= 3, learning_rate=1)
#
# train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X_train, y_train,
#                    train_sizes=np.linspace(0.1, 1.0, 10),
#                    cv=10, scoring=None,
#                    exploit_incremental_learning=False,
#                    n_jobs=1, pre_dispatch="all", verbose=0)
# train_scores =np.average(train_scores,axis=1)
# test_scores = np.average(test_scores, axis=1)
#
# # i = 1
# plt.figure()
# plt.title('Adult Boosting: Accuracy vs. training_size')
# plt.plot(train_sizes_abs, train_scores, '-', label='Training score')
# plt.plot(train_sizes_abs, test_scores, 'x-', label='Cross-Validation score')
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('training_size')
# plt.ylabel('Accuracy')
# plt.show()
#
#
#
# estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9, min_samples_split = 5), n_estimators= 3, learning_rate=1)
# estimator.fit(X_train, y_train)
# y_pred = estimator.predict(X_test)
# acc_score = accuracy_score(y_test, y_pred, normalize=True)
# print acc_score
#
#
# SVM
# training_accuracy = []
# validation_accuracy = []
# test_accuracy = []
# scoring = {'acc': 'accuracy',
#            'prec_macro': 'precision_macro',
#            'rec_micro': 'recall_macro'}
# # clf = svm.LinearSVC()
# clf = svm.SVC(C = 1.0, kernel='rbf', decision_function_shape='ovo', gamma=1.1)
# scores = cross_validate(clf, X_train, y_train, cv=10, scoring=scoring, return_train_score=True)
# validation_accuracy.append(np.average(scores['train_acc']))
# test_accuracy.append(np.average(scores['test_acc']))
#
# print np.average(test_accuracy)
# print np.average(validation_accuracy)

# #
# estimator = svm.SVC(C = 1.0, kernel='rbf', decision_function_shape='ovo', gamma=1.1)
#
# train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X_train, y_train,
#                    train_sizes=np.linspace(0.1, 1.0, 10),
#                    cv=10, scoring=None,
#                    exploit_incremental_learning=False,
#                    n_jobs=1, pre_dispatch="all", verbose=0)
# train_scores =np.average(train_scores,axis=1)
# test_scores = np.average(test_scores, axis=1)
#
# # i = 1
# plt.figure()
# plt.title('Adult SVM(linear): Accuracy vs. training_size')
# plt.plot(train_sizes_abs, train_scores, '-', label='Training score')
# plt.plot(train_sizes_abs, test_scores, 'x-', label='Cross-Validation score')
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('training_size')
# plt.ylabel('Accuracy')
# plt.show()
# #
# #
# #
# estimator = svm.SVC(C = 1.0, kernel='rbf', decision_function_shape='ovo', gamma=1.1)
# estimator.fit(X_train, y_train)
# y_pred = estimator.predict(X_test)
# acc_score = accuracy_score(y_test, y_pred, normalize=True)
# print acc_score

# neural network
# training_accuracy = []
# validation_accuracy = []
# test_accuracy = []
# layer_values = range(1, 10)
# scoring = {'acc': 'accuracy',
#            'prec_macro': 'precision_macro',
#            'rec_micro': 'recall_macro'}
# for n in layer_values:
#     hiddens = tuple(n * [20])
#     clf = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=hiddens, random_state=1)
#     clf.fit(X_train, y_train)
#     scores = cross_validate(clf, X_train, y_train, cv=5, scoring=scoring, return_train_score=True)
#     validation_accuracy.append(np.average(scores['train_acc']))
#     test_accuracy.append(np.average(scores['test_acc']))
#
# #
# Kvals = range(1, 10)
# plt.figure()
# plt.title('Adult Neural Network: Accuracy vs. hidden_layer_sizes')
# plt.plot(Kvals, validation_accuracy, '-', label='train_acc')
# plt.plot(Kvals, test_accuracy, 'x-', label='validation_acc')
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('hidden_layer_sizes')
# plt.ylabel('Accuracy')
# plt.show()
#
#
# #
# hiddens = tuple(7*[20])
# estimator = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=hiddens, random_state=1)
#
# train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X_train, y_train,
#                    train_sizes=np.linspace(0.1, 1.0, 10),
#                    cv=5, scoring=None,
#                    exploit_incremental_learning=False,
#                    n_jobs=1, pre_dispatch="all", verbose=0)
# train_scores =np.average(train_scores,axis=1)
# test_scores = np.average(test_scores, axis=1)
#
# # i = 1
# plt.figure()
# plt.title('Adult Neural Network: Accuracy vs. training_size')
# plt.plot(train_sizes_abs, train_scores, '-', label='Training score')
# plt.plot(train_sizes_abs, test_scores, 'x-', label='Cross-Validation score')
# plt.grid(True)
# plt.legend(loc='best')
# plt.xlabel('training_size')
# plt.ylabel('Accuracy')
# plt.show()

# hiddens = tuple(7*[20])
# estimator = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=hiddens, random_state=1)
# estimator.fit(X_train, y_train)
# y_pred = estimator.predict(X_test)
# acc_score = accuracy_score(y_test, y_pred, normalize=True)
# print acc_score

