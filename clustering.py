from sklearn.cluster import KMeans, DBSCAN, k_means
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.spatial import distance_matrix

def main():
    #save data as numpy arrays
    data1 = np.loadtxt('dataset1.txt')
    data2 = np.loadtxt('dataset2.txt')

    clusternums = [2,3,4,5]
    #kmeans clustering with both datasets determining SSE with k on intererval 2 to 5
    for clusternum in clusternums:
        #k_means returns the final value of the inertia criterion (sum of squared distances to the closest centroid for
        #all observations in the training set).
        centroid1, label1, sse1 = k_means(data1, n_clusters=clusternum)
        print("For n_clusters =", clusternum,
              "The SSE for data set 1 is :", sse1)
        centroid2, label2, sse2 = k_means(data2, n_clusters=clusternum)
        print("For n_clusters =", clusternum,
              "The SSE for data set 2 is :", sse2)

    #best KMeans clustering for both datasets (minimalized inertia) k = 5
    kmeans1 = label1
    kmeans2 = label2

    #dbscan clustering with both datasets
    dbscan1 = DBSCAN(eps=9, min_samples=5).fit_predict(data1)
    dbscan2 = DBSCAN(eps=.28, min_samples=2).fit_predict(data2)

    #upper triangle of proximity matrix (X)
    iu = np.triu_indices(750,1)
    proximity1 = distance_matrix(data1, data1)
    proximity2 = distance_matrix(data2, data2)
    prox1 = proximity1[iu]
    prox2 = proximity2[iu]

    #upper triangle of incidence matrices (Y)
    kincident1 = np.zeros((750, 750))
    kincident2 = np.zeros((750, 750))
    dbincident1 = np.zeros((750, 750))
    dbincident2 = np.zeros((750, 750))
    for i in range(750):
        for j in range(750):
            if(kmeans1[i] == kmeans1[j]):
                kincident1[i][j] = 1
            else:
                kincident1[i][j] = 0
            if (kmeans2[i] == kmeans2[j]):
                kincident2[i][j] = 1
            else:
                kincident2[i][j] = 0
            if (dbscan1[i] == dbscan1[j]):
                dbincident1[i][j] = 1
            else:
                dbincident1[i][j] = 0
            if (dbscan2[i] == dbscan2[j]):
                dbincident2[i][j] = 1
            else:
                dbincident2[i][j] = 0
    kinc1 = kincident1[iu]
    kinc2 = kincident2[iu]
    dbinc1 = dbincident1[iu]
    dbinc2 = dbincident2[iu]

    #mean of X
    mean1 = np.mean(prox1)
    mean2 = np.mean(prox2)

    #mean of Y
    km1 = np.mean(kinc1)
    km2 = np.mean(kinc2)
    dm1 = np.mean(dbinc1)
    dm2 = np.mean(dbinc2)

    #standard deviation of X
    std1 = np.std(prox1)
    std2 = np.std(prox2)

    #standard deviation of Y
    ksdy1 = np.std(kinc1)
    ksdy2 = np.std(kinc2)
    dbsdy1 = np.std(dbinc1)
    dbsdy2 = np.std(dbinc2)

    #covariance-> sum((X - u)(Y - v))/n
    kcov1 = (np.sum((prox1-mean1)*(kinc1-km1))/280875)
    kcov2 = (np.sum((prox2-mean2)*(kinc2-km2))/280875)
    dbcov1 = (np.sum((prox1-mean1)*(dbinc1-dm1))/280875)
    dbcov2 = (np.sum((prox2-mean2)*(dbinc2-dm2))/280875)

    #correlation scores -> covariance(X,Y)/(stdev(X)*stdev(Y))4
    kcorr1 = (kcov1/(std1*ksdy1))
    kcorr2 = (kcov2/(std2*ksdy2))
    dbcorr1 = (dbcov1/(std1*dbsdy1))
    dbcorr2 = (dbcov2/(std2*dbsdy2))

    print("The kmeans correlation for data set 1 is :", kcorr1)
    print("The kmeans correlation for data set 2 is :", kcorr2)
    print("The DBSCAN correlation for data set 1 is :", dbcorr1)
    print("The DBSCAN correlation for data set 2 is :", dbcorr2)

    #Sillouette scores
    ksilhouette1 = silhouette_score(data1, kmeans1)
    ksilhouette2 = silhouette_score(data2, kmeans2)
    dbsilhouette1 = silhouette_score(data1, dbscan1)
    dbsilhouette2 = silhouette_score(data2, dbscan2)

    print("The kmeans average silhouette_score for data set 1 is :", ksilhouette1)
    print("The kmeans average silhouette_score for data set 2 is :", ksilhouette2)
    print("The DBSCAN average silhouette_score for data set 1 is :", dbsilhouette1)
    print("The DBSCAN average silhouette_score for data set 2 is :", dbsilhouette2)


if __name__ == '__main__':
    main()