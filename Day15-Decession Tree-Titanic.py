# from Decession tree we have to predict wherether the passengers has survived or not
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("D:/CDAC-AI/Training Material/Machine Learning/Class Codes/titanic.csv")
print(data)

print(data.shape)
print(data.groupby('Survived').size())

a=pd.get_dummies(data['Sex'])
data1=pd.concat([data,a],axis=1)
print(data1)

data1['Age'].fillna(value=data['Age'].mean(), inplace=True)
from sklearn.model_selection import train_test_split
data1.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data1.loc[:, data1.columns != 'Survived'], data1['Survived'], stratify=data1['Survived'],random_state=42)#stratify tells on which column we have to do training and testing 
print(y_train.value_counts())
print(y_test.value_counts())


from sklearn.tree import DecisionTreeClassifier
feature_name=list(X_train.columns)
class_name=list(y_train.unique())
print(feature_name)
print(class_name)

from sklearn import metrics
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
 
#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn import tree
#plt.figure(figsize=(70,10))
tree.plot_tree(clf,filled=True)
plt.show()

from sklearn.model_selection import GridSearchCV#what the best criteria
gd = GridSearchCV(clf,{'max_depth':[3,4,5,6,7,8,9],'criterion':['gini','entropy']},cv=8)
gd=gd.fit(X_train,y_train)
gd.best_params_
gd.best_score_
y_pred=gd.predict(X_test)
print("Accuracy : ",metrics.accuracy_score(y_test,y_pred))


#BY PCA algorithm
print("BY PCA algorithm:")
from sklearn.decomposition import PCA
pca=PCA()

pca.fit(data1)
pca.explained_variance_ratio_
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum ())
plt.figure(figsize=(10,5))
plt.plot (range (1,9), pca.explained_variance_ratio_.cumsum (), marker = 'o', linestyle = '--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
