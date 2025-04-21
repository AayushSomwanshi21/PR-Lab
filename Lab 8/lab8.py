import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


img = cv2.imread('Lab 8/Screenshot (2).png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (128, 128))
pixel_data = img.reshape((-1, 3))

kmeans = KMeans(n_clusters=5)
kmeans.fit(pixel_data)

clustered_img = kmeans.cluster_centers_[kmeans.labels_]
clustered_img = clustered_img.reshape(img.shape).astype('uint8')
plt.imshow(clustered_img)
plt.show()
