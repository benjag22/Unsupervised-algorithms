#pragma once
#include <bits/stdc++.h>
#include "truncate.h"

struct DBPoint {
    char name;
    double x;
    double y;
    int cluster = -1;  // no visitado:-1, ruido:-2 , >= 0 = número de cluster
    bool visited = false;

    DBPoint(char n, double x_coord, double y_coord)
            : name(n), x(x_coord), y(y_coord) {}
};

class DBScan {
private:
    std::vector<DBPoint> points;
    double eps;
    int minPoints;
    int currentCluster;
    std::vector<std::vector<double>> distanceMatrix;
    std::vector<char> clusterSymbols;

    double euclideanDistance(const DBPoint& p1, const DBPoint& p2) const {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return truncate(sqrt(dx * dx + dy * dy), 2);
    }

    void buildDistanceMatrix() {
        int n = points.size();
        distanceMatrix.assign(n, std::vector<double>(n, 0.0));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    distanceMatrix[i][j] = 0.0;
                } else {
                    distanceMatrix[i][j] = euclideanDistance(points[i], points[j]);
                }
            }
        }
    }

    std::vector<int> getNeighbors(int pointIndex) const {
        std::vector<int> neighbors;
        for (int i = 0; i < points.size(); ++i) {
            if (distanceMatrix[pointIndex][i] <= eps) {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    }

    void expandCluster(int pointIndex, std::vector<int>& neighbors) {
        points[pointIndex].cluster = currentCluster;

        for (int i = 0; i < neighbors.size(); ++i) {
            int neighborIndex = neighbors[i];

            if (!points[neighborIndex].visited) {
                points[neighborIndex].visited = true;
                std::vector<int> newNeighbors = getNeighbors(neighborIndex);

                if (newNeighbors.size() >= minPoints) {
                    for (int newNeighbor : newNeighbors) {
                        bool alreadyInList = false;
                        for (int existing : neighbors) {
                            if (existing == newNeighbor) {
                                alreadyInList = true;
                                break;
                            }
                        }
                        if (!alreadyInList) {
                            neighbors.push_back(newNeighbor);
                        }
                    }
                }
            }

            if (points[neighborIndex].cluster == -1) {
                points[neighborIndex].cluster = currentCluster;
            }
        }
    }

public:
    DBScan(double epsilon, int minPts)
            : eps(epsilon), minPoints(minPts), currentCluster(0) {
        clusterSymbols = {'X', 'O', 'D', 'S', 'T', 'P', 'Q', 'R'};
    }

    void addPoint(char name, double x, double y) {
        points.emplace_back(name, x, y);
    }

    void run() {
        if (points.empty()) return;

        buildDistanceMatrix();

        std::cout << "=== ALGORITMO DBSCAN ===" << std::endl;
        std::cout << "Parametros: eps = " << eps << ", minPoints = " << minPoints << std::endl;

        for (auto& point : points) {
            point.visited = false;
            point.cluster = -1;
        }
        currentCluster = 0;

        for (int i = 0; i < points.size(); ++i) {
            if (points[i].visited) continue;

            points[i].visited = true;
            std::vector<int> neighbors = getNeighbors(i);

            if (neighbors.size() < minPoints) {
                points[i].cluster = -2;
            } else {
                expandCluster(i, neighbors);
                currentCluster++;
            }
        }
        printResults();
        drawGrid();
    }

    void runWithDifferentEps(double newEps) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "CAMBIANDO EPS A " << newEps << std::endl;

        double oldEps = eps;
        eps = newEps;

        for (auto& point : points) {
            point.visited = false;
            point.cluster = -1;
        }
        currentCluster = 0;

        std::cout << "Nuevos parametros: eps = " << eps << ", minPoints = " << minPoints << std::endl;

        for (int i = 0; i < points.size(); ++i) {
            if (points[i].visited) continue;

            points[i].visited = true;
            std::vector<int> neighbors = getNeighbors(i);

            if (neighbors.size() < minPoints) {
                points[i].cluster = -2; // Marcar como ruido
            } else {
                expandCluster(i, neighbors);
                currentCluster++;
            }
        }
        printResults();
        drawGrid();
    }

    void printResults() const {
        std::cout << "\nResultados del clustering:" << std::endl;
        std::set<int> uniqueClusters;
        int noisePoints = 0;

        for (const auto& point : points) {
            if (point.cluster == -2) {
                noisePoints++;
            } else if (point.cluster >= 0) {
                uniqueClusters.insert(point.cluster);
            }
        }

        std::cout << "Numero de clusters encontrados: " << uniqueClusters.size() << std::endl;
        std::cout << "Numero de puntos de ruido: " << noisePoints << std::endl;
        std::cout << std::endl;

        for (int clusterId : uniqueClusters) {
            std::cout << "Cluster " << clusterId << ": ";
            bool first = true;
            for (const auto& point : points) {
                if (point.cluster == clusterId) {
                    if (!first) std::cout << ", ";
                    std::cout << point.name << "(" << point.x << "," << point.y << ")";
                    first = false;
                }
            }
            std::cout << std::endl;
        }

        if (noisePoints > 0) {
            std::cout << "Ruido: ";
            bool first = true;
            for (const auto& point : points) {
                if (point.cluster == -2) {
                    if (!first) std::cout << ", ";
                    std::cout << point.name << "(" << point.x << "," << point.y << ")";
                    first = false;
                }
            }
            std::cout << std::endl;
        }
    }

    void drawGrid() const {
        std::cout << "\nMatriz 10x10 con clusters:" << std::endl;
        std::cout << "   ";
        for (int x = 0; x <= 10; ++x) {
            std::cout << std::setw(2) << x;
        }
        std::cout << std::endl;

        for (int y = 10; y >= 0; --y) {
            std::cout << std::setw(2) << y << " ";
            for (int x = 0; x <= 10; ++x) {
                char symbol = '.';

                for (const auto &point: points) {
                    if ((int) point.x == x && (int) point.y == y) {
                        if (point.cluster == -2) {
                            symbol = 'N';
                        } else if (point.cluster >= 0) {
                            symbol = clusterSymbols[point.cluster % clusterSymbols.size()];
                        } else {
                            symbol = '?';
                            break;
                        }
                    }

                    std::cout << symbol << " ";
                }
                std::cout << std::endl;
            }

            std::cout << "Leyenda: ";
            std::set<int> uniqueClusters;
            for (const auto &point: points) {
                if (point.cluster >= 0) {
                    uniqueClusters.insert(point.cluster);
                }
            }

            for (int clusterId: uniqueClusters) {
                std::cout << clusterSymbols[clusterId % clusterSymbols.size()]
                          << " Cluster " << clusterId << ", ";
            }
            std::cout << "N Ruido" << std::endl;
        }
    }
};