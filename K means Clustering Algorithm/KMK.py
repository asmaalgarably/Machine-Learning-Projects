from sklearn.cluster import KMeans
import pandas as pd

data = pd.DataFrame({
    'study_hours': [2, 3, 4, 5, 6, 7, 8, 9],
    'grade': [50, 55, 60, 65, 70, 75, 85, 90]
})
print(data)


model = KMeans(n_clusters=3, random_state=44)
model.fit(data)
data['cluster'] = model.labels_
print(model.inertia_)
print(data)
