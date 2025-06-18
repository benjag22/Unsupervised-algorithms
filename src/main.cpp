#include <bits/stdc++.h>
#include "k-means.h"

int main() {
    std::cout << "=== Ejercicio 1 ===" << std::endl;

    Kmeans kmeans(3);

    kmeans.addPoint('A', 2, 10);
    kmeans.addPoint('B', 2, 5);
    kmeans.addPoint('C', 8, 4);
    kmeans.addPoint('D', 5, 8);
    kmeans.addPoint('E', 7, 5);
    kmeans.addPoint('F', 6, 4);
    kmeans.addPoint('G', 1, 2);
    kmeans.addPoint('H', 4, 9);

    kmeans.printEuclideanMatrix();

    std::vector<Centroid> initialCentroids1 = {
            Centroid(2, 10, 'A'), // Cluster 0
            Centroid(5, 8, 'D'),  // Cluster 1
            Centroid(1, 2, 'G')   // Cluster 2
    };
    kmeans.setInitialCentroids(initialCentroids1);

    kmeans.run(3);
    return 0;
}
