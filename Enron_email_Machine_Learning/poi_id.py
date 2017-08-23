#!/usr/bin/python

import sys
import pickle
import numpy as np
from tester import test_classifier
from sklearn import tree
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


sys.path.append("../tools/")


### Task 1: Select features.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'exercised_stock_options', 'total_stock_value']

financial_features = ['salary', 'deferral_payments', 'total_payments', 
                      'loan_advances', 'bonus', 'restricted_stock_deferred', 
                      'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 
                  'from_messages', 'from_this_person_to_poi', 
                  'shared_receipt_with_poi']

 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict

#Set all NaN values to zero
for name in my_dataset.keys():
    for feature in my_dataset[name]:
        if my_dataset[name][feature] == 'NaN':
            my_dataset[name][feature] = 0


#remove invalid entry 'TOTAL'
my_dataset.pop('TOTAL')


#remove invalid entry 'THE TRAVEL AGENCY IN THE PARK'
my_dataset.pop('THE TRAVEL AGENCY IN THE PARK')


'''Calculates a fraction based on the given num and denom'''
def calcFraction(num, denom):
    if num == 0 or denom == 0:
        fraction = 0
    else:
        fraction =  float(num) / float(denom)
    return round(fraction, 2)


#Compute fraction of emails to poi
for name in my_dataset.keys():
    my_dataset[name]['frac_to_poi'] = \
    calcFraction(my_dataset[name]['from_this_person_to_poi'], \
                 my_dataset[name]['from_messages'])

##Compute fraction of emails from poi 
for name in my_dataset.keys():
    my_dataset[name]['frac_from_poi'] = \
    calcFraction(my_dataset[name]['from_poi_to_this_person'], \
                 my_dataset[name]['to_messages'])


#Get metrics of data set
entry_count = len(my_dataset)
feature_count = len(my_dataset['METTS MARK'])

poi_count = 0 
for name in my_dataset.keys():
    if my_dataset[name]['poi'] == 1:
        poi_count +=1

nonpoi_count = entry_count - poi_count

print("Stats: total entries {}, number of features {}, poi {}, non-poi {}" \
      .format(entry_count, feature_count, poi_count, nonpoi_count))
  

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(np.array(features))
features = scaled_features
folds = 1000
cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )           
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

        
### Task 4: Experiment with varity of classifiers

#Initial test of GaussianNB no custom features
print("\n")
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
nb_score = metrics.accuracy_score(pred, labels_test)
nb_precision = metrics.precision_score(pred, labels_test)
nb_recall = metrics.recall_score(pred, labels_test)
print("defalut nb accuracy {}, precision {}, recall {}".format(round(nb_score, 2), nb_precision, nb_recall))


#Initial untuned test of SVM no custom features
clf = svm.SVC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
svm_score = metrics.accuracy_score(pred, labels_test)
svm_precision = metrics.precision_score(pred, labels_test)
svm_recall = metrics.recall_score(pred, labels_test)
print("defalut svm accuracy {}, precision {}, recall {}".format(round(svm_score, 2), svm_precision, svm_recall))


#Initial untuned test of DecisionTree no custom features
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
dt_score = metrics.accuracy_score(pred, labels_test)
dt_precision = metrics.precision_score(pred, labels_test)
dt_recall = metrics.recall_score(pred, labels_test)
print("defalut dt accuracy {}, precision {}, recall {}".format(round(dt_score, 2), dt_precision, dt_recall))


#set features list to include the two custome features 
features_list = ['poi','salary', 'bonus', 'exercised_stock_options', 
                 'total_stock_value','frac_to_poi', 'frac_from_poi']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(np.array(features))
features = scaled_features
folds = 1000
cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )


#Initial test of GaussianNB with custom features
print("\n")
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
nb_score = metrics.accuracy_score(pred, labels_test)
nb_precision = metrics.precision_score(pred, labels_test)
nb_recall = metrics.recall_score(pred, labels_test)
print("Custom feature nb accuracy {}, precision {}, recall {}".format(round(nb_score, 2), nb_precision, nb_recall))


#Initial untuned test of SVM with custom features
clf = svm.SVC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
svm_score = metrics.accuracy_score(pred, labels_test)
svm_precision = metrics.precision_score(pred, labels_test)
svm_recall = metrics.recall_score(pred, labels_test)
print("Custom feature svm accuracy {}, precision {}, recall {}".format(round(svm_score, 2), svm_precision, svm_recall))


#Initial untuned test of DecisionTree with custom features
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
dt_score = metrics.accuracy_score(pred, labels_test)
dt_precision = metrics.precision_score(pred, labels_test)
dt_recall = metrics.recall_score(pred, labels_test)
print("Custom feature dt accuracy {}, precision {}, recall {}".format(round(dt_score, 2), dt_precision, dt_recall))
print("DT feature importance {}".format(clf.feature_importances_))


### Task 5: Tune classifier to achieve better than .3 precision and recall based on testing scrip.


#Exploring tuning of SVM via GridSearchCV
from sklearn.metrics import precision_score, make_scorer

#parameters to test
parameters = {'C':[1,10], 'gamma': [1,10]}

#scorer to use
precision_scorer = make_scorer(precision_score)

#Test and out put optimal results
svc = svm.SVC()
grid = GridSearchCV(svc, param_grid=parameters, scoring=precision_scorer) 
grid.fit(features_train, labels_train)
print("\nBest params for svm {}".format(grid.best_params_))


#Testing SVM with optimal perameters against tester.test_classifier
clf = svm.SVC(kernel = 'rbf', C = 1, gamma = 1)     
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
svm_score = metrics.accuracy_score(pred, labels_test)
svm_precision = metrics.precision_score(y_pred = pred, y_true = labels_test)
svm_recall = metrics.recall_score(y_pred = pred, y_true = labels_test)

print("\nSVM prformace according to tester.test_classifier")
test_classifier(clf, my_dataset, features_list)


#exploring tuning of dt via GridSearchCV
#parameters to test
param = {'min_samples_split': [2,3,4,5,6,7,8,9,10]}

#scorer to use
precision_scorer = make_scorer(precision_score)

#test and output optimal results
dt = tree.DecisionTreeClassifier()
grid = GridSearchCV(dt, param_grid = param, scoring=precision_scorer) 
grid.fit(features_train, labels_train)
print("\nBest params for dt {}".format(grid.best_params_))


#Testing DT with optimal perameters against tester.test_classifier
clf = tree.DecisionTreeClassifier(min_samples_split = 2)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
dt_score = metrics.accuracy_score(pred, labels_test)
dt_precision = metrics.precision_score(y_pred = pred, y_true = labels_test)
dt_recall = metrics.recall_score(y_pred = pred, y_true = labels_test)

print("\ndt prformace according to tester.test_classifier")
test_classifier(clf, my_dataset, features_list)


#Checking which features are most important to the DT classifier
print(clf.feature_importances_)


#Final Selection GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
nb_score = metrics.accuracy_score(pred, labels_test)
nb_precision = metrics.precision_score(pred, labels_test)
nb_recall = metrics.recall_score(pred, labels_test)
print("\nGaussianNB prformace according to tester.test_classifier")
test_classifier(clf, my_dataset, features_list)


### Task 6: Dump classifier, dataset, and features_list so anyone can
### check results.

dump_classifier_and_data(clf, my_dataset, features_list)
