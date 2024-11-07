#Simple Linear Regression


# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the dataset

dataset = pd.read_csv('C:/Users/Admin/Desktop/SimpleLinearRegression.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
model=regressor.fit(X_train, y_train)

print('intercept:', model.intercept_)
print('slope:', model.coef_)


# Predicting the Test set results

y_pred = model.predict(X_test)


# Visualising the Training set results

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
r_sq = model.score(X_train, y_train)
print('train set resudial of determination:', r_sq)



# Visualising the Test set results

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
r_sq = model.score(X_test, y_test)
print('test set resudial of determination:', r_sq)