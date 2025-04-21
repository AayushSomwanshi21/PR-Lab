import matplotlib.pyplot as plt
from fcmeans import FCM
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

wine = load_wine()

x = wine.data[:, :2]

scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)

fcm = FCM(n_clusters=3)
fcm.fit(x_scaled)

centres = fcm.centers

labels = fcm.predict(x_scaled)

plt.figure(figsize=(8, 6))

for i in range(3):
    plt.scatter(x_scaled[labels == i, 0],
                x_scaled[labels == i, 1], label=f'Cluster {i + 1}')

plt.scatter(centres[:, 0], centres[:, 1],
            color='black', marker='x', label='Centres', s=200)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
