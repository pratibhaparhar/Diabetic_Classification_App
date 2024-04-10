from sklearn.naive_bayes import GaussianNB
from sklearn import metrics as m
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

diabetes_df=pd.read_csv('artifacts\diabetes.csv')
# print(diabetes_df)
# print(diabetes_df.info())
# print(diabetes_df.isnull().sum())


x=diabetes_df.drop('Outcome',axis=1)
y=diabetes_df[['Outcome']]
# print(x)
# print(y)

# To Check dataset is balanaced or not
# print(y.value_counts(normalize=True))

# splittting data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.70,random_state=40)
# print(x_test.shape,
# x_train.shape,
# y_test.shape,
# y_train.shape)

# Model Training
GNB=GaussianNB()
GNB.fit(x_train,y_train)

# Predication
y_pred=GNB.predict(x_test)

# Evaluation of model
# print(m.accuracy_score(y_test,y_pred))
# print(m.classification_report(y_test,y_pred))

# creating pickel  file
pickle.dump(GNB,open('navie_bayes_dia_model.pkl','wb'))