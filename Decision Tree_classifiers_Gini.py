"""
Created on Sat Sep 10 17:27:00 2022
"""

#=======================================================================
import pandas as pd
df = pd.read_csv("Sales.csv")  
df.shape
df.info()

# Label encode
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

    
# pip install conda

# if we use gini index max depth = 6 is better
# if we use entropy max depth = 7 is better

# from sklearn import tree
# import graphviz
# dot_data = tree.export_graphviz(classifier, out_file=None, 
#                     filled=True, rounded=True,  
#                     special_characters=True)  
# graph = graphviz.Source(dot_data)  
# graph

# import graphviz.backend as be
# cmd = ["dot", "-V"]
# stdout, stderr = be.run(cmd)
# print( stderr )

# pip install dtreeviz
# import dtreeviz

# viz = dtreeviz(classifier.fit(X_train,Y_train),
#               Y_train,
#               Y_pred_train, fancy=False )


######################################################

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
Classifier=DecisionTreeClassifier(max_depth=7)

Bag=BaggingClassifier(base_estimator=Classifier,max_samples=0.6,n_estimators=100)

Bag.fit(X,Y)
Y_pred_bag=Bag.predict(X)
Bagging_error = accuracy_score(Y,Y_pred_bag)
    
#################################################    

from sklearn.ensemble import RandomForestClassifier
Regressor=RandomForestClassifier(max_depth=7)
RF=RandomForestClassifier(max_features=0.4,n_estimators=500)
RF.fit(X,Y)
Y_pred_RF=RF.predict(X)
RF_error = accuracy_score(Y,Y_pred_RF)


#######################################################

from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(n_estimators=500,learning_rate=5)

GB.fit(X,Y)
Y_pred_GB = GB.predict(X)
GB_error = accuracy_score(Y,Y_pred_GB)






















