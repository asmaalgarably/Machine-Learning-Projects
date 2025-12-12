# calling the modules and libraries
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
#load a datasets in kaggle
datasets=pd.read_csv(r"C:\Users\ACER\Desktop\MLProject\SupportVectorRegression\housing.csv")
# Apply One-Hot Encoding to categorical features and drop the first dummy variable
datasets=pd.get_dummies(datasets,drop_first=True)
# split the data to x and y to train the model 
x=datasets.iloc[:,:-1]
y=datasets.iloc[:,-1]
#class simpleImputer to fill any empty data in cloumn
imp=SimpleImputer(missing_values=np.nan,strategy=np.mean)
x=imp.fit_transform(x)
# splitting data to test and train
x_train, x_test,y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=44)
# standardscaler to standard a range data
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
# gradSearchcv to choses best parameters
param_grid={'C':[.1,1,5,10,100,1000],
            'gamma':[0.01,0.001,0.0001],
            'kernel':['rbf']}
# the model 
model=SVR()
#applied the gridserachcv on model 
grid=GridSearchCV(estimator=model,verbose=3,n_jobs=-1,cv=5,param_grid=param_grid)
grid.fit(x_train,y_train)
y_pred=grid.predict(x_test)
# evaluation
mae=mean_absolute_error(y_test,y_pred)
print(mae)
print(grid.best_params_)




