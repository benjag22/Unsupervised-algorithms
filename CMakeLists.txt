cmake_minimum_required(VERSION 3.30)
project(Unsupervised_algorithms)

set(CMAKE_CXX_STANDARD 20)

add_executable(Unsupervised_algorithms src/main.cpp
        src/k-means.h
        src/DBscan.h
        src/truncate.h
        src/truncate.cpp)
