import pandas as pd
import numpy as np
from rich import reconfigure
from sklearn.ensemble import RandomForestClassifier

#import data

df=pd.read_csv("Data/diabetes.csv")

#define feautuer metrices x and y
x=df.drop(['Outcome'],axis=1)
x.to_csv("train.csv")
y=df['Outcome']
x.to_csv("test.csv")
#train test spllit
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25, random_state = 18)

#import model
lr=RandomForestClassifier(n_estimators=53,n_jobs=1,random_state=8)
# fit model
lr.fit(x,y)

#lets serialize the model
import joblib

joblib.dump(lr,'randomfs.pkl')
print("Random forest Model saved ")

#lets load the model
lr=joblib.load('randomfs.pkl')

#save features from training
rnd_columns=list(X_train.columns)
joblib.dump(rnd_columns,'rnd_columns.pkl')
print("Random Forst Model Columns saved")
