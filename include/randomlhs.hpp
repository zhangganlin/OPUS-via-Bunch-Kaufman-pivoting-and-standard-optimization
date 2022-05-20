#include <vector>
#include <algorithm>
#include <stdio.h>

template <class T>
bool findranksCompare(const std::pair<T, int> first, const std::pair<T, int> second);

template <class T>
void findorder_zero(const std::vector<T> & v, std::vector<int> & order);

template <class T>
void findorder(const std::vector<T> & v, std::vector<int> & order);


double randfrom(double min, double max);


void randomLHS(int n, int k, double * matrix, double range_lo, double range_hi);
    