from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
data=load_breast_cancer()
x=data.data
y=data.target
scler = StandardScaler()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44)
x_train = scler.fit_transform(x_train)
x_test=scler.transform(x_test)
model =LogisticRegression(max_iter=2000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)

