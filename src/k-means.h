#pragma once
#include <bits/stdc++.h>

double truncate(double number_val, int n)
{
    bool negative = false;
    if (number_val == 0) {
        return 0;
    } else if (number_val < 0) {
        number_val = -number_val;
        negative = true;
    }
    double pre_digits = std::log10(number_val) + 1;
    if (pre_digits < 17) {
        double post_digits = 17 - pre_digits;
        double factor = std::pow(10, post_digits);
        number_val = std::round(number_val * factor) / factor;
        factor = std::pow(10, n);
        number_val = std::trunc(number_val * factor) / factor;
    } else {
        number_val = std::round(number_val);
    }
    if (negative) {
        number_val = -number_val;
    }
    return number_val;
}

struct Point {
    char name;
    double x;
    double y;
    int cluster = -1;

    Point(char n, double x_coord, double y_coord) : name(n), x(x_coord), y(y_coord) {}
};

struct Centroid {
    double x;
    double y;
    char originalName;

    Centroid(double x_coord, double y_coord, char name = ' ')
            : x(x_coord), y(y_coord), originalName(name) {}
};

class Kmeans{
    private:
        std::vector<Point> points;
        std::vector<Centroid> centroids;
        int k;
        std::vector<char> clusterSymbols;

        [[nodiscard]] static double euclideanDistance(const Point& point, const Centroid& centroid) {
            double dx = point.x - centroid.x;
            double dy = point.y - centroid.y;
            return sqrt(dx * dx + dy * dy);
        }

        void assignPointsToClusters() {
            for (auto& point : points) {
                double minDistance = std::numeric_limits<double>::max();
                int bestCluster = 0;

                for (int i = 0; i < centroids.size(); ++i) {
                    double distance = euclideanDistance(point, centroids[i]);
                    if (distance < minDistance) {
                        minDistance = distance;
                        bestCluster = i;
                    }
                }
                point.cluster = bestCluster;
            }
        }

        void updateCentroids() {
            std::vector<Centroid> newCentroids;
            std::vector<int> clusterCounts(k, 0);
            std::vector<double> sumX(k, 0.0), sumY(k, 0.0);

            for (const auto& point : points) {
                if (point.cluster >= 0 && point.cluster < k) {
                    sumX[point.cluster] += point.x;
                    sumY[point.cluster] += point.y;
                    clusterCounts[point.cluster]++;
                }
            }

            for (int i = 0; i < k; ++i) {
                if (clusterCounts[i] > 0) {
                    double newX = truncate(sumX[i] / clusterCounts[i], 2);
                    double newY = truncate(sumY[i] / clusterCounts[i], 2);
                    newCentroids.emplace_back(newX, newY);
                } else {
                    newCentroids.push_back(centroids[i]);
                }
            }

            centroids = newCentroids;
        }

    public:

        explicit Kmeans(int numClusters) : k(numClusters) {
            clusterSymbols = {'X', 'O', 'D'};
        }

        void addPoint(char name, double x, double y) {
            points.emplace_back(name, x, y);
        }

        void setInitialCentroids(const std::vector<Centroid>& initialCentroids) {
            centroids = initialCentroids;
        }

        void runIteration() {
            assignPointsToClusters();
            updateCentroids();
        }

        void run(int iterations) {
            std::cout << "ALGORITMO K-MEANS" << std::endl;
            printInitialCentroids();

            for (int iter = 1; iter <= iterations; ++iter) {
                runIteration();
                printResults(iter);
            }
        }

        void printInitialCentroids() const {
            std::cout << "Centroides iniciales:" << std::endl;
            for (int i = 0; i < centroids.size(); ++i) {
                std::cout << "Cluster " << i << ": " << centroids[i].originalName
                          << "(" << centroids[i].x << ", " << centroids[i].y << ")" << std::endl;
            }
        }

        void printClusters(int iteration) const {
            std::cout << "\n=== ITERACION " << iteration << " ===" << std::endl;
            std::cout << "a) Nuevos clusters:" << std::endl;

            for (int i = 0; i < k; ++i) {
                std::cout << "Cluster " << i << ": ";
                bool first = true;
                for (const auto& point : points) {
                    if (point.cluster == i) {
                        if (!first) std::cout << ", ";
                        std::cout << point.name << "(" << point.x << "," << point.y << ")";
                        first = false;
                    }
                }
                std::cout << std::endl;
            }
        }

        void printCentroids() const {
            std::cout << "b) Nuevos centroides:" << std::endl;
            for (int i = 0; i < centroids.size(); ++i) {
                std::cout << "Centroide " << i << ": (" << centroids[i].x
                          << ", " << centroids[i].y << ")" << std::endl;
            }
        }

        void drawGrid(int iteration) const {
            std::cout << "c) Matriz de 10x10 (Iteracion " << iteration << "):" << std::endl;
            std::cout << "   ";
            for (int x = 0; x <= 10; ++x) {
                std::cout << std::setw(2) << x;
            }
            std::cout << std::endl;

            for (int y = 10; y >= 0; --y) {
                std::cout << std::setw(2) << y << " ";
                for (int x = 0; x <= 10; ++x) {
                    char symbol = '.';

                    for (const auto& point : points) {
                        if ((int)point.x == x && (int)point.y == y) {
                            symbol = clusterSymbols[point.cluster % 3];
                            break;
                        }
                    }

                    for (const auto & centroid : centroids) {
                        if (abs(centroid.x - x) < 0.5 && abs(centroid.y - y) < 0.5) {
                            symbol = 'C'; // C para centroide
                            break;
                        }
                    }

                    std::cout << symbol << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "Leyenda: X Cluster 0, O Cluster 1, D Cluster 2, C Centroide" << std::endl;
        }

        void printResults(int iteration) const {
            printClusters(iteration);
            printCentroids();
            drawGrid(iteration);
            std::cout << "\n" << std::string(50, '=') << std::endl;
        }

        void printEuclideanMatrix() const {
            int n = points.size();
            std::cout << "Matriz de distancias Euclidianas:" << std::endl;

            std::cout << "    ";
            for (const auto& point : points) {
                std::cout << std::setw(6) << point.name;
            }
            std::cout << std::endl;

            for (int i = 0; i < n; ++i) {
                std::cout << std::setw(3) << points[i].name << " ";
                for (int j = 0; j < n; ++j) {
                    if (i == j) {
                        std::cout << std::setw(6) << "0";
                    } else {
                        double dx = points[i].x - points[j].x;
                        double dy = points[i].y - points[j].y;
                        double distance = truncate(sqrt(dx * dx + dy * dy), 2);
                        std::cout << std::setw(6) << distance;
                    }
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    };