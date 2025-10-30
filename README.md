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



<img width="348" height="295" alt="image" src="https://github.com/user-attachments/assets/a703e7d4-3f16-41c2-a2af-342d89771946" />


<img width="276" height="156" alt="image" src="https://github.com/user-attachments/assets/a2fbff28-3515-4a90-916f-e35f9fb95a42" />



<img width="502" height="336" alt="image" src="https://github.com/user-attachments/assets/43036478-84d2-4b56-abb8-2e6600bf52a1" />



<img width="530" height="118" alt="image" src="https://github.com/user-attachments/assets/2754ef2c-c0f8-4ecf-8abe-3f749b247715" />



<img width="1112" height="454" alt="image" src="https://github.com/user-attachments/assets/8aa2cddf-5e68-4099-a908-2fa4d15dca58" />



<img width="1082" height="576" alt="image" src="https://github.com/user-attachments/assets/c1763de4-28eb-40e7-800a-d3c34dbf14ad" />







## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
