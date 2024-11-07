import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age','label']

pima = pd.read_csv('E:/WINTER SEM 2018-19/LAB material/classification_EXP_2/diabetes.csv', header=None, names=col_names)

pima.head()

# define X and y
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']

# X is a matrix, hence we use [] to access the features we want in feature_cols
X = pima[feature_cols]
print(X)
# y is a vector, hence we use dot to access 'label'
y = pima.label
print(y)
# split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=47,train_size = 0.8)

#Importing the Decision tree classifier from the sklearn library.
#from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = 'entropy')

#Training the decision tree classifier. 
clf.fit(X_train, y_train)

#Predicting labels on the test set.
y_pred =  clf.predict(X_test)

# calculate accuracy
#from sklearn import metrics
print('Accuracy Score on test data: ',metrics.accuracy_score(y_test, y_pred))

# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
print(metrics.confusion_matrix(y_test, y_pred))

# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, y_pred)
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

# use float to perform true division, not integer division
print((TP + TN) / float(TP + TN + FP + FN))
print('Accuracy Score on confusion_matrix: ' ,metrics.accuracy_score(y_test, y_pred))


classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)
print('Error Score on confusion_matrix: ' ,1 - metrics.accuracy_score(y_test, y_pred))

sensitivity = TP / float(FN + TP)
print(sensitivity)
print('sensitivity Score on confusion_matrix: ',metrics.recall_score(y_test, y_pred))

specificity = TN / (TN + FP)
print(specificity)
print('specificity Score on confusion_matrix: ', specificity)

precision = TP / float(TP + FP)
print(precision)
print('precision Score on confusion_matrix: ',metrics.precision_score(y_test, y_pred))
