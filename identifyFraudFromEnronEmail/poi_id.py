#!/usr/bin/python

import sys
import pickle
import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

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
#create new features "from_poi_to_this_person_ratio" and "from_this_person_to_poi_Ratio"
for f in data_dict:
    if data_dict[f]["from_poi_to_this_person"]!="NaN":
        data_dict[f]["from_poi_to_this_person_ratio"] = 1.0 * data_dict[f]["from_poi_to_this_person"]/data_dict[f]["to_messages"]
    else:
        data_dict[f]["from_poi_to_this_person_ratio"] = "NaN"
        
for f in data_dict:
    if data_dict[f]["from_this_person_to_poi"]!="NaN":
        data_dict[f]["from_this_person_to_poi_Ratio"] = 1.0 * data_dict[f]["from_this_person_to_poi"]/data_dict[f]["from_messages"]
    else:
        data_dict[f]["from_this_person_to_poi_Ratio"] = "NaN"   
        
#make the featurs_list
features_list  = ['poi','total_payments','total_stock_value',
                  'salary', 'deferral_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
                  'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
                  'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
                  'shared_receipt_with_poi','from_poi_to_this_person_ratio','from_this_person_to_poi_Ratio']

#data_array = featureFormat(data_dict,features_list)
#labels, features = targetFeatureSplit(data_array)

#split the data into training and testing
"""
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,labels, test_size = 0.3, random_state = 42)
"""
### Store to my_dataset for easy export below.

df = pd.DataFrame.from_dict(data_dict,orient="index")
df = df.replace("NaN",np.nan)
df = df[features_list]
print df.shape
df.head()


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)