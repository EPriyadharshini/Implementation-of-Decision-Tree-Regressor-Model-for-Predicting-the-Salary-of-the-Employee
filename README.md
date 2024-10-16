# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array. 8.Apply to new unknown values.


## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: Priyadharshini E 
RegisterNumber:  212223230159

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

dt.predict([[5,6]])plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()


```



### Output:



![Screenshot 2024-09-30 160218](https://github.com/user-attachments/assets/45c88a84-c7a0-4abc-8880-a31cd901bd99)


![Screenshot 2024-09-30 160243](https://github.com/user-attachments/assets/4132b1c0-27db-4381-b133-d8289100e573)



![Screenshot 2024-09-30 160300](https://github.com/user-attachments/assets/6d4242de-f158-4c19-8552-f66afb22a57a)



![Screenshot 2024-09-30 160324](https://github.com/user-attachments/assets/827a0d30-5d11-46d1-97c3-5523684df9b9)



![Screenshot 2024-09-30 160344](https://github.com/user-attachments/assets/b2fbc657-b24e-4f29-bf38-a646f9cfd6dc)



![Screenshot 2024-09-30 160411](https://github.com/user-attachments/assets/d7b7da85-5458-4338-a983-7251a5d3c9a4)



![image](https://github.com/user-attachments/assets/f4fa14a7-89e4-4c16-a70b-20261465cd76)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
