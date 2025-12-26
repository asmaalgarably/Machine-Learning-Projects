from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data=load_breast_cancer()
print(data.feature_names)
x=data.data
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44)

model=GaussianNB()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print('Training', model.score(x_train, y_train))
print("Testing",model.score(x_test,y_test))

CM=confusion_matrix(y_test,y_pred)
print(CM)

sns.heatmap(CM,center=True)
plt.show()