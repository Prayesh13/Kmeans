from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
from kmeans import KMeans

iris_df = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")


# we take two most important columns in iris dataset.(i.e.sepal_width,petal_length)
X = iris_df.iloc[:,1:3].values

# Create a demo dataset using sklearn : -

# centroid = [(-5,-5),(5,5),(-5,5),(5,-5)]
# cluster_std = [1,1,1,1]
#
# X,y = make_blobs(n_samples=100, cluster_std=cluster_std,centers=centroid, n_features=2,random_state=2)

km = KMeans(n_clusters=3,max_iter=100)
y_pred = km.fit_predict(X)
#
#
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.scatter(km.centroid[:,0],km.centroid[:,1],s=50,c='red')
plt.show()