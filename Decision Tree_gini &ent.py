"""
Created on Sat Sep 10 17:27:00 2022
"""
import pandas as pd
#=======================================================================
df = pd.read_csv("Sales.csv")  
df.shape
df.info()
list(df.head())

# Label encode
# import sys
# from sklearnex import patch_sklearn


from sklearn.preprocessing import LabelEncoder 
LE = LabelEncoder()
df['ShelveLoc'] = LE.fit_transform(df['ShelveLoc'])
df['Urban'] = LE.fit_transform(df['Urban'])
df['US'] = LE.fit_transform(df['US'])
df.head()




X = df.iloc[:,1:11]
list(X)
Y = df['high']

# split the data in to train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, random_state=42)


# model fitting
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier(criterion='gini',max_depth=None) 
classifier.fit(X_train,Y_train)


classifier.tree_.max_depth # number of levels
classifier.tree_.node_count # counting the number of nodes

Y_pred_train = classifier.predict(X_train)
Y_pred_test = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
Training_accuracy = accuracy_score(Y_train,Y_pred_train).round(2)
Test_accuracy = accuracy_score(Y_test,Y_pred_test).round(2)

print("Training_accuracy",Training_accuracy)
print("Test_accuracy",Test_accuracy)


# for loop
training_acc = []
test_acc = []

for i in range(1,13):
    classifier = DecisionTreeClassifier(criterion='gini',max_depth=i) 
    classifier.fit(X_train,Y_train)
    Y_pred_train = classifier.predict(X_train)
    Y_pred_test = classifier.predict(X_test)
    training_acc.append(accuracy_score(Y_train,Y_pred_train).round(2))
    test_acc.append(accuracy_score(Y_test,Y_pred_test).round(2))
    
    
print(training_acc)
print(test_acc)

# finalized the max_depth at 7
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier(criterion='entropy',max_depth=7) 
classifier.fit(X_train,Y_train)


classifier.tree_.max_depth # number of levels
classifier.tree_.node_count # counting the number of nodes

Y_pred_train = classifier.predict(X_train)
Y_pred_test = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
Training_accuracy = accuracy_score(Y_train,Y_pred_train).round(2)
Test_accuracy = accuracy_score(Y_test,Y_pred_test).round(2)

print("Training_accuracy",Training_accuracy)
print("Test_accuracy",Test_accuracy)

    
    

# if we use gini index max depth = 6 is better
# if we use entropy max depth = 7 is better
import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(classifier , out_file = None ,
                                filled = True , rounded = True ,
                                special_characters = True)
graph = graphviz.Source(dot_data)
graph














