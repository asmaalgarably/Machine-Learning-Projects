import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\ACER\Desktop\MLProject\Niave Bayes\heart.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44)
sclar=StandardScaler()
x_train=sclar.fit_transform(x_train)
x_test=sclar.transform(x_test)

model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("Training",model.score(x_train,y_train))
print("Testing",model.score(x_test,y_test))

CM=confusion_matrix(y_test,y_pred)

sns.heatmap(CM,center=True)
plt.show()