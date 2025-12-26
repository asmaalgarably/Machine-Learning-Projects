import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.DataFrame({
    'free':     [1, 1, 0, 0, 0, 0],
    'win':      [1, 0, 0, 0, 1, 0],
    'money':   [1, 1, 0, 0, 1, 0],
    'meeting': [0, 0, 1, 1, 0, 1],
    'label':   ['spam', 'spam', 'not_spam', 'not_spam', 'spam', 'not_spam']
})

# print(data)

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44)

model=BernoulliNB(alpha=1.0,binarize=1)
model.fit(x_train,y_train)
y_pred=model.predict([[1,1,0,0]])
print(y_pred)

# print("Training",model.score(x_train,y_train))
# print("Testing",model.score(x_test,y_test))

# cm=confusion_matrix(y_test,y_pred)

# sns.heatmap(cm,center=True)
# plt.show()
