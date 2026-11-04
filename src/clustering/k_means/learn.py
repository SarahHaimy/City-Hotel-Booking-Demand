from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

model = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
labels = model.fit_predict(X_scaled)

print(model.cluster_centers_)   # centroids
print(model.inertia_)           # objective value