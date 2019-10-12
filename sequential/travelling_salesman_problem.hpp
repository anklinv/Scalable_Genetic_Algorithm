#include <vector>
#include "city.hpp"

#ifndef DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP
#define DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP

using namespace std;

class TravellingSalesmanProblem {
public:
    TravellingSalesmanProblem(const int problem_size, const int population_count);
    int problem_size;
    int population_count;
    vector<City> cities;
    int** population;
};


#endif //DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP
