from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from fcmeans import FCM
from sklearn.preprocessing import StandardScaler


data = load_wine()

x = data.data[:, :2]

scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)

fcm = FCM(n_clusters=3)
fcm.fit(x_scaled)

centres = fcm.centers
labels = fcm.predict(x_scaled)

plt.figure(figsize=(10, 10))

for i in range(3):
    plt.scatter(x_scaled[labels == i, 0], x_scaled[labels == i, 1], s=100)

plt.scatter(centres[:, 0], centres[:, 1], s=200, marker='x', color='black')
plt.show()
