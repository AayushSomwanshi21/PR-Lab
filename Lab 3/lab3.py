import numpy as np
from scipy.spatial.distance import mahalanobis


def get_input(num):
    '''
     sample input
     poly1 : 10.0 20.0, 30.0 40.0, 50.0 60.0
     poly2 : 100.0 200.0, 300.0 400.0, 500.0 600.0
    '''
    print(f'Enter coordinates for polygon {num}')
    points = input('Enter coordinates: ').strip().split(',')
    polygon = []

    for point in points:

        x, y = point.strip().split()
        polygon.append((float(x), float(y)))

    return polygon


def compute_mahalanobis(poly1, poly2):

    poly1 = np.array(poly1)
    poly2 = np.array(poly2)

    centroid1 = np.mean(poly1, axis=0)
    centroid2 = np.mean(poly2, axis=0)
    print(centroid1, centroid2)

    combined = np.vstack((centroid1, centroid2))

    cov = np.cov(combined.T)

    try:
        inv_cov = np.linalg.inv(cov)

    except np.linalg.LinAlgError:
        print('Cannot inverted singular matrix')

    distance = mahalanobis(centroid1, centroid2, inv_cov)

    return distance


if __name__ == '__main__':

    try:
        poly1 = get_input(1)
        poly2 = get_input(2)
        print(poly1, poly2)
        distace = compute_mahalanobis(poly1, poly2)
        print('Mahanolobis Distance: ', distace)
    except ValueError as e:
        print(e)
