#include "randomlhs.hpp"

template <class T>
bool findranksCompare(const std::pair<T, int> first, const std::pair<T, int> second)
{
    return (first.first < second.first);
}

template <class T>
void findorder_zero(const std::vector<T> & v, std::vector<int> & order)
{
    // create a vector of pairs to hold the value and the integer rank
    std::vector<std::pair<T, int> > p(v.size());
    
    typename std::vector<T>::const_iterator vi;
    typename std::vector<std::pair<T, int> >::iterator pi;
    int position = 0;
    for (vi = v.begin(), pi = p.begin();
            vi != v.end() && pi != p.end(); ++vi, ++pi)
    {
        *pi = std::pair<T, int>(*vi, position);
        position++;
    }

    // if the rank vector is not the right size, resize it (the original values may be lost)
    if (order.size() != v.size())
    {
        order.resize(v.size());
    }

    // sort the pairs of values
    std::sort(p.begin(), p.end(), findranksCompare<double>);

    // take the ranks from the pairs and put them in the rank vector
    std::vector<int>::iterator oi;
    for (oi = order.begin(), pi = p.begin(); 
            oi != order.end() && pi != p.end(); ++oi, ++pi)
    {
        *oi = pi->second;
        //order[i] = p[i].second;
    }
}

template <class T>
void findorder(const std::vector<T> & v, std::vector<int> & order)
{
    findorder_zero<T>(v, order);
    for (std::vector<int>::size_type i = 0; i < order.size(); i++)
    {
        order[i] += 1;
    }
}




void randomLHS(int n, int k, double ** matrix,  double range_lo, double range_hi)
{   
    double range = range_hi - range_lo;
    std::vector<int> orderVector = std::vector<int>(n);
    std::vector<double> randomunif1 = std::vector<double>(n);
    for (int jcol = 0; jcol < k; jcol++)
    {
        for (int irow = 0; irow < n; irow++)
        {
            randomunif1[irow] = (rand()/(double)RAND_MAX);
        }
        findorder<double>(randomunif1, orderVector);
        for (int irow = 0; irow < n; irow++)
        {
            matrix[irow][jcol] = orderVector[irow];
            matrix[irow][jcol] = matrix[irow][jcol]/n*range+range_lo;
        }
    }
}
    