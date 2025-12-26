from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data=load_breast_cancer()
x=data.data
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=44)

model=MultinomialNB(alpha=1.0)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("Training",model.score(x_train,y_train))
print("Testing",model.score(x_test,y_test))

cm=confusion_matrix(y_test,y_pred)

print(cm)
sns.heatmap(cm,center=True)
plt.show()
