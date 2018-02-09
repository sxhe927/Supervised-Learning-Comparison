# The following code is meant to do supervised learing for the Wine Quality dataset
# Be sure all the libraries and packets are installed
# Five learning algorithms are implemented one by one, it is recommeded to run each of them and comment the others
#


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# load all the data from the csv file

df = pd.read_csv('winequality-white.csv', delimiter=';', quotechar='"')
df['quality'] = np.where(df["quality"] >= 7, 1, 0)
X = df.ix[:, df.columns != 'quality']
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

plt.hist(y_test, range=(0,1))
plt.hist(y_train, range=(0,1))
plt.xlabel('Quality')
plt.xticks([0, 1])
plt.ylabel('Count')
plt.title('Distribution of Wine Quality Classes')
plt.show()


# # decision trees


training_accuracy = []
validation_accuracy = []
test_accuracy = []
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}
for i in xrange(2,20):
    classifier = DecisionTreeClassifier(max_depth=i, min_samples_split = 5)
    classifier.fit(X_train, y_train)
    # print classifier
    t_a = accuracy_score(y_train, classifier.predict(X_train))
    training_accuracy.append(t_a)
    scores = cross_validate(classifier, X_train, y_train, cv = 10, scoring=scoring, return_train_score=True)
    validation_accuracy.append(np.average(scores['train_acc']))
    test_accuracy.append(np.average(scores['test_acc']))

Kvals = range(2, 20)
plt.figure()
plt.title('Wine Quality Decision Tree: Accuracy vs. max_depth')
plt.plot(Kvals, validation_accuracy, '-', label='train_acc')
plt.plot(Kvals, test_accuracy, 'x-', label='validation_acc')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.show()

#

estimator = DecisionTreeClassifier(max_depth=15, min_samples_split = 5)

train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X_train, y_train,
                   train_sizes=np.linspace(0.1, 1.0, 10),
                   cv=10, scoring=None,
                   exploit_incremental_learning=False,
                   n_jobs=1, pre_dispatch="all", verbose=0)
train_scores =np.average(train_scores,axis=1)
test_scores = np.average(test_scores, axis=1)

# i = 1
plt.figure()
plt.title('Wine Quality Decision Tree: Accuracy vs. training_size')
plt.plot(train_sizes_abs, train_scores, '-', label='Training score')
plt.plot(train_sizes_abs, test_scores, 'x-', label='Cross-Validation score')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('training_size')
plt.ylabel('Accuracy')
plt.show()



estimator = DecisionTreeClassifier(max_depth=15, min_samples_split = 5)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
acc_score = accuracy_score(y_test, y_pred, normalize=True)
print acc_score



# boosting

training_accuracy = []
validation_accuracy = []
test_accuracy = []
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}

for j in xrange(1,40):
    AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_split=5), n_estimators=j, learning_rate = 1)
    AdaBoost.fit(X_train, y_train)
    scores = cross_validate(AdaBoost, X_train, y_train, cv=10, scoring=scoring, return_train_score=True)
    validation_accuracy.append(np.average(scores['train_acc']))
    test_accuracy.append(np.average(scores['test_acc']))


Kvals = range(1, 40)
plt.figure()
plt.title('Wine Quality Boosting: Accuracy vs. n_estimators')
plt.plot(Kvals, validation_accuracy, '-', label='train_acc')
plt.plot(Kvals, test_accuracy, 'x-', label='validation_acc')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('n_estimator')
plt.ylabel('Accuracy')
plt.show()




estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_split = 5), n_estimators= 15, learning_rate=1)

train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X_train, y_train,
                   train_sizes=np.linspace(0.1, 1.0, 10),
                   cv=10, scoring=None,
                   exploit_incremental_learning=False,
                   n_jobs=1, pre_dispatch="all", verbose=0)
train_scores =np.average(train_scores,axis=1)
test_scores = np.average(test_scores, axis=1)

# i = 1
plt.figure()
plt.title('Wine Quality Boosting: Accuracy vs. training_size')
plt.plot(train_sizes_abs, train_scores, '-', label='Training score')
plt.plot(train_sizes_abs, test_scores, 'x-', label='Cross-Validation score')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('training_size')
plt.ylabel('Accuracy')
plt.show()



estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_split = 5), n_estimators= 15, learning_rate=1)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
acc_score = accuracy_score(y_test, y_pred, normalize=True)
print acc_score
#





# KNN

training_accuracy = []
validation_accuracy = []
test_accuracy = []
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}


for k in xrange(1,30):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    t_a = accuracy_score(y_train, knn.predict(X_train))
    training_accuracy.append(t_a)
    scores = cross_validate(knn, X_train, y_train, cv=10, scoring=scoring, return_train_score=True)
    validation_accuracy.append(np.average(scores['train_acc']))
    test_accuracy.append(np.average(scores['test_acc']))


Kvals = range(1, 30)
plt.figure()
plt.title('Wine Quality KNN: Accuracy vs. n_neighbors')
plt.plot(Kvals, validation_accuracy, '-', label='train_acc')
plt.plot(Kvals, test_accuracy, 'x-', label='validation_acc')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()




estimator = KNeighborsClassifier(n_neighbors=2)

train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X_train, y_train,
                   train_sizes=np.linspace(0.1, 1.0, 10),
                   cv=10, scoring=None,
                   exploit_incremental_learning=False,
                   n_jobs=1, pre_dispatch="all", verbose=0)
train_scores =np.average(train_scores,axis=1)
test_scores = np.average(test_scores, axis=1)

# i = 1
plt.figure()
plt.title('Wine Quality KNN: Accuracy vs. training_size')
plt.plot(train_sizes_abs, train_scores, '-', label='Training score')
plt.plot(train_sizes_abs, test_scores, 'x-', label='Cross-Validation score')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('training_size')
plt.ylabel('Accuracy')
plt.show()



estimator = KNeighborsClassifier(n_neighbors=2)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
acc_score = accuracy_score(y_test, y_pred, normalize=True)
print acc_score
#




# SVM

training_accuracy = []
validation_accuracy = []
test_accuracy = []
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}
clf = svm.LinearSVC()
# clf = svm.SVC(C = 1.0, kernel='rbf', decision_function_shape='ovo', gamma=1.1)
clf.fit(X_train, y_train)
scores = cross_validate(clf, X_train, y_train, cv=10, scoring=scoring, return_train_score=True)
validation_accuracy.append(np.average(scores['train_acc']))
test_accuracy.append(np.average(scores['test_acc']))

print np.average(test_accuracy)
print np.average(validation_accuracy)

#
estimator = svm.SVC(C = 1.0, kernel='rbf', decision_function_shape='ovo', gamma=1.1)

train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X_train, y_train,
                   train_sizes=np.linspace(0.1, 1.0, 10),
                   cv=10, scoring=None,
                   exploit_incremental_learning=False,
                   n_jobs=1, pre_dispatch="all", verbose=0)
train_scores =np.average(train_scores,axis=1)
test_scores = np.average(test_scores, axis=1)

# i = 1
plt.figure()
plt.title('Wine Quality SVM(rbf): Accuracy vs. training_size')
plt.plot(train_sizes_abs, train_scores, '-', label='Training score')
plt.plot(train_sizes_abs, test_scores, 'x-', label='Cross-Validation score')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('training_size')
plt.ylabel('Accuracy')
plt.show()



estimator = svm.SVC(C = 1.0, kernel='rbf', decision_function_shape='ovo', gamma=1.1)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
acc_score = accuracy_score(y_test, y_pred, normalize=True)
print acc_score




# neural network

training_accuracy = []
validation_accuracy = []
test_accuracy = []
layer_values = range(10)
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}
for n in layer_values:
    hiddens = tuple(n * [25])
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hiddens, random_state=1)
    clf.fit(X_train, y_train)
    scores = cross_validate(clf, X_train, y_train, cv=10, scoring=scoring, return_train_score=True)
    validation_accuracy.append(np.average(scores['train_acc']))
    test_accuracy.append(np.average(scores['test_acc']))

#
Kvals = range(10)
plt.figure()
plt.title('Wine Quality Neural Network: Accuracy vs. hidden_layer_sizes')
plt.plot(Kvals, validation_accuracy, '-', label='train_acc')
plt.plot(Kvals, test_accuracy, 'x-', label='validation_acc')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('hidden_layer_sizes')
plt.ylabel('Accuracy')
plt.show()



hiddens = tuple(5*[25])
estimator = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hiddens, random_state=1)

train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X_train, y_train,
                   train_sizes=np.linspace(0.1, 1.0, 10),
                   cv=10, scoring=None,
                   exploit_incremental_learning=False,
                   n_jobs=1, pre_dispatch="all", verbose=0)
train_scores =np.average(train_scores,axis=1)
test_scores = np.average(test_scores, axis=1)

# i = 1
plt.figure()
plt.title('Wine Quality Neural Network: Accuracy vs. training_size')
plt.plot(train_sizes_abs, train_scores, '-', label='Training score')
plt.plot(train_sizes_abs, test_scores, 'x-', label='Cross-Validation score')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('training_size')
plt.ylabel('Accuracy')
plt.show()

hiddens = tuple(5*[25])
estimator = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hiddens, random_state=1)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
acc_score = accuracy_score(y_test, y_pred, normalize=True)
print acc_score
