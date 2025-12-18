import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error

data=pd.read_csv(r'C:\Users\ACER\Desktop\MLProject\K-Nearest Neighbors\data_Regression.csv',encoding='latin1')
# print(data.columns)

x=data['text']
y=data['hours']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44)

vector=TfidfVectorizer()
x_train_vector=vector.fit_transform(x_train)
x_test_vector=vector.transform(x_test)

model=KNeighborsRegressor(n_neighbors=2,weights='distance')
model.fit(x_train_vector,y_train)
y_pred=model.predict(x_test_vector)
mae=mean_absolute_error(y_test,y_pred)
print(mae)

q = ['مادة البرمجة المتقدمة']
qv=vector.transform(q)
p=model.predict(qv)
print(p)

