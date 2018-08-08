#!/usr/bin/python

import sys
import pickle
import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import tester

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import BaggingClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Explore the data dictonary, check the number of keys, features, an example data
#point, and the number and percentage of poi
def exploration_data(data_dict):
    total_people = len(data_dict)
    print "total number of people in the dataset: ", total_people

    number_features = len(data_dict["ALLEN PHILLIP K"])
    total_features = data_dict["ALLEN PHILLIP K"].keys()
    print "Total features in the dataset: "
    print total_features
    print "poi is the label, the number of all other featuers in the dataset is: ", number_features-1
    print "An example entry 'ALLEN PHILLIP K' in the dataset: "
    print data_dict["ALLEN PHILLIP K"]

    #check for the poi
    total_poi = 0
    for k in data_dict:
        if data_dict[k]["poi"] == True:
            total_poi+=1
    print "The number of poi:", total_poi
    print "The percentage of poi is {:0.2f}%.".format(100.00*total_poi/total_people)

exploration_data(data_dict)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)

### Task 3: Create new feature(s)
#feature engineering
#remove feature "email_address"
#create new features "from_poi_to_this_person_ratio" and "from_this_person_to_poi_ratio"
for f in data_dict:
    if data_dict[f]["from_poi_to_this_person"]!="NaN":
        data_dict[f]["from_poi_to_this_person_ratio"] = 1.0 * data_dict[f]["from_poi_to_this_person"]/data_dict[f]["to_messages"]
    else:
        data_dict[f]["from_poi_to_this_person_ratio"] = "NaN"

for f in data_dict:
    if data_dict[f]["from_this_person_to_poi"]!="NaN":
        data_dict[f]["from_this_person_to_poi_ratio"] = 1.0 * data_dict[f]["from_this_person_to_poi"]/data_dict[f]["from_messages"]
    else:
        data_dict[f]["from_this_person_to_poi_ratio"] = "NaN"

#make the featurs_list
features_list  = ['poi','total_payments','total_stock_value',
                  'salary', 'deferral_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
                  'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                  'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi','from_poi_to_this_person_ratio','from_this_person_to_poi_ratio']


#To further explore the dataset, transform the dictionary into a dataframe
df = pd.DataFrame.from_dict(data_dict,orient="index")
df = df.replace("NaN",np.nan)
df = df[features_list]
print "This dataset has {} rows and {} columns".format(df.shape[0],df.shape[1])

#impute null financial features by 0
df.loc[:,"total_payments":"director_fees"] = df.loc[:,"total_payments":"director_fees"].fillna(0)

#impute null contact features by the mean
df.loc[:,"to_messages":"from_this_person_to_poi_ratio"] = df.loc[:,"to_messages":"from_this_person_to_poi_ratio"].apply(lambda x: x.fillna(x.mean()),axis = 0)

#Check incorrect data point in "total_payments" and "total_stock_value"
#Check the "total_payments"
payments = ["salary","deferral_payments","loan_advances","bonus","deferred_income",
            "expenses","other","director_fees","long_term_incentive"]
df[df[payments].sum(axis = "columns")!=df.total_payments]
#Correct total_payments
df["total_payments"]["BELFER ROBERT"] = df[payments].loc["BELFER ROBERT"].sum()
df["total_payments"]["BHATNAGAR SANJAY"] = df[payments].loc["BHATNAGAR SANJAY"].sum()

#Check the "total_stock_value"
stock_value = ["restricted_stock_deferred","exercised_stock_options","restricted_stock"]
df[df[stock_value].sum(axis = "columns") !=df.total_stock_value]

#Correct total_stock_value
df["total_stock_value"]["BELFER ROBERT"] = df[stock_value].loc["BELFER ROBERT"].sum()
df["total_stock_value"]["BHATNAGAR SANJAY"] = df[stock_value].loc["BHATNAGAR SANJAY"].sum()

#final check the dataset
df.head()

### Store to my_dataset for easy export below.
my_dataset = df.to_dict(orient = "index")
feature_list_new = list(df.columns.values)
print "There are {} people and {} features in this dataset".format(len(my_dataset),len(feature_list_new))

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset,feature_list_new,sort_keys=True)
labels,features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#select features using random forest
clf = RandomForestClassifier(random_state=0)
tester.test_classifier(clf, my_dataset,feature_list_new)

#list the feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

features_list = sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_),
                           feature_list_new[1:]), reverse=True)

features_df = pd.DataFrame(features_list)
features_df = features_df.rename(index = str, columns = {0:"importance",1:"feature"})

#based on feature importance, select following features for modeling 
#the threshold of accuracy is 0.08 and 0.06
selected_features1 = ['poi','other','from_this_person_to_poi_ratio','exercised_stock_options','expenses']

selected_features3 = ['poi','other','from_this_person_to_poi_ratio','exercised_stock_options','expenses',
                     'salary','restricted_stock']

selected_features2 = ['poi','other','from_this_person_to_poi_ratio','exercised_stock_options','expenses',
                     'salary']

#Try a variatye of classifiers
#gaussian navie bays with different features
nb = GaussianNB()
for i in [selected_features1,selected_features2,selected_features3]:
    tester.test_classifier(nb, my_dataset,i)

#decision tree with different features
dt = DecisionTreeClassifier(random_state=0)
for i in [selected_features1,selected_features2,selected_features3]:
    tester.test_classifier(dt,my_dataset,i)

#random forest
rf = RandomForestClassifier(random_state=0)
for i in [selected_features1,selected_features2,selected_features3]:
    tester.test_classifier(rf,my_dataset,i)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#From task 4, decision_tree performed the best, here I will tune parameters for
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

# tune decision tree
data = featureFormat(my_dataset, selected_features1, sort_keys=True)
labels, features = targetFeatureSplit(data)

# 1000 folds are used to make it as similar as possible to tester.py.
folds = 1000
decision_tree = DecisionTreeClassifier(random_state=0)
dt_parameters = {'criterion':('gini','entropy'),\
                 'min_samples_split':(5,10,15,20),\
                 'max_depth':(5,7,10,20)}
# store the split instance into cv and use it in the GridSearchCV.
cv = StratifiedShuffleSplit(labels, folds,random_state=0)
grid = GridSearchCV(decision_tree, dt_parameters, cv=cv, scoring='f1')
grid.fit(features, labels)

print("The best parameters are %s with a score of %0.4f"
      %(grid.best_params_, grid.best_score_))
#The best parameters are {'min_samples_split': 20, 'criterion': 'entropy', 'max_depth': 7} with a score of 0.4265

#decision tree model with optimal parametres
dt_best = DecisionTreeClassifier(min_samples_split=20,max_depth=7,criterion='entropy',random_state=0)
tester.test_classifier(dt_best,my_dataset,selected_features1)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(dt_best, my_dataset, selected_features1)
tester.main()
