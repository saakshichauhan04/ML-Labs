import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/Admin/Desktop/insurance.csv")
df.head(5)
plt.scatter(df.age,df.charges,color= 'red', marker='+')
plt.xlabel("Age of person")
plt.ylabel("Bought Insurance 1=Bought 0=Did not Buy")
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
#Split the dataset into train and test sets (70:30)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10)
print(x_test)
from sklearn.linear_model import LogisticRegression
reg =LogisticRegression()
reg.fit(x_train, y_train)
yPrediction = reg.predict(x_test) #Predict the test set
print(yPrediction)

plt.scatter(x_test,y_test,color='green', marker='*')
plt.scatter(x_test,yPrediction,color='blue', marker='.')
ins_accuracy=reg.score(x_test,y_test)
print('insurance score:',ins_accuracy*100)
o=reg.predict_proba(x_test)
print(o)

yPrediction1=reg.predict(96.9)
print(yPrediction1)
