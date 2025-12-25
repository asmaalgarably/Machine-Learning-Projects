import pandas as pd 
from sklearn.cluster import KMeans

data = pd.DataFrame({'glucose': [90, 110, 130, 160, 180, 200, 220, 250],
                    'bmi': [22, 24, 26, 29, 31, 33, 35, 38],
                     'age': [25, 30, 35, 40, 45, 50, 55, 60]})

print(data)

model=KMeans(n_clusters=3,random_state=44)
model.fit(data)
data['cluster'] = model.labels_

print(model.inertia_)
print(data)
print(model.cluster_centers_)
