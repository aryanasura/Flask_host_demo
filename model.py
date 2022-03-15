import pandas as pd
import numpy as np 
from sklearn import linear_model as lm
from word2number import w2n
import math
import pickle
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("hiring.csv")

df.experience = df.experience.fillna('zero')
df.experience = df.experience.apply(w2n.word_to_num)

score = math.floor(df['test_score'].mean())
df['test_score'] = df['test_score'].fillna(score)

reg = lm.LinearRegression()

x=df[['experience','test_score','interview_score']]
y=df['salary($)']

df.reset_index(drop=True, inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.1)

reg.fit(X_train,y_train)

pred=reg.predict(X_test)
pred.astype(float)

pred.reshape(-1, 1)

pickle.dump(reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))