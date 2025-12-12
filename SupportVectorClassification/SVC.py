import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
# load dataset
dataset=load_breast_cancer()
x = dataset.data
y = dataset.target

# splitting data for train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44)

# to standralization the range dataset
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
smo=SMOTE(random_state=44)
x_train_smo,y_train_smo=smo.fit_resample(x_train,y_train)

param_grid={'C':[10,100,500,1000,5000,10000],
            'gamma':[0.1,0.01,0.001,0.0001],
           'kernel': ['rbf'], 'class_weight': ['balanced']}
#model
model=SVC()
#Gridsearchcv for the best parameters 
grid=GridSearchCV(estimator=model,n_jobs=-1,cv=5,verbose=3,param_grid=param_grid)
grid.fit(x_train_smo,y_train_smo)
#Euvalution
y_pred=grid.predict(x_test)
cr=classification_report(y_test,y_pred)
print(cr)
print(grid.best_params_)

#saving the model
joblib.dump(grid.best_estimator_,'SupportVectorClassification\SVC model.pkl')
joblib.dump(scaler,'SupportVectorClassification\scaler.pkl')


