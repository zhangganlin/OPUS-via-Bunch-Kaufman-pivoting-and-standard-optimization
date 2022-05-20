#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

void read_matrix(double* A, int m, int n, string path){
    std::ifstream in(path);
    std::string line;
    
    int i = 0, k = 0;

    while (std::getline(in, line))
    {
        double value;
        int k = 0;
        std::stringstream ss(line);

        while (ss >> value)
        {
            A[i*n+k] = value;
            ++k;
        }
        ++i;
    }
}
