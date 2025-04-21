from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits = load_digits()
x, y = digits.data, digits.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

svc = SVC(kernel='linear', C=1.0, random_state=42)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(x_pca)

plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1],
            c=kmeans.labels_, cmap="tab10", s=30)
plt.title("K-Means Clustering of MNIST Digits")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.show()
