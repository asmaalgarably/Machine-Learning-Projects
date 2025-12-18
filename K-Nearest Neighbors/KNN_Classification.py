import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv(
    r'C:\Users\ACER\Desktop\MLProject\K-Nearest Neighbors\TestData.csv', encoding='latin1')
x = data['text']
y=data['label']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=44)

vector=TfidfVectorizer()
x_vector_train=vector.fit_transform(x_train)
x_vector_test=vector.transform(x_test)


model=KNeighborsClassifier(n_neighbors=2,weights='distance',metric='cosine')
model.fit(x_vector_train,y_train)

y_pred=model.predict(x_vector_test)
cr=classification_report(y_test,y_pred)
print(cr)

#test
q = ['كم عدد الساعات لمادة الشبكات']
qv =vector.transform(q)
knn_pred=model.predict(qv)
print(knn_pred)
