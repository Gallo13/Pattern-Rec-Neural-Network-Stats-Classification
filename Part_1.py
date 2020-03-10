# Created: 2/19/2020
# Last Modified: 2/19/2020
# Pattern Recognition and Neural Networks
# Descriptive Statistics, Classification and Analysis Using Python and Python Libraries
# Part 1

# Listing 1a
# Imports for libraries
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import random

# Listing 1b
# Dataset
filename = "data_banknote_authentication.txt"
names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
dataSetCsv = pd.read_csv(filename,',',error_bad_lines=False, names=names)
dataset = pd.DataFrame(dataSetCsv)
# dataset = read_csv(filename, names=names, delimiter='\t')

# Listing 2a
# Descriptive statistics
print(dataset.shape)

# Listing 2b
print(dataset.head(20))

# Listing 2c
print(dataset.describe())

# Listing 2d
print(dataset.groupby('class').size())

# Listing 3a
# Box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
pyplot.show()

# Listing 3b
# Histograms
dataset.hist()
pyplot.show()

# Listing 3c
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# Listing 4a
# Split-out validation dataset
array = dataset.values
random.shuffle(array)
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Listing 4b
# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Listing 4c
# Compare algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparision')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_ticklabels(names)
pyplot.show()

# Listing 5
# Make predictions on validation dataset
'''
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
'''

for name, model in models:
    print("------------------------------------------------")
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    print("------------------------------------------------")
