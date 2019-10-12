#include <cstdlib>
#include <random>
#include "travelling_salesman_problem.hpp"
#include "city.hpp"

TravellingSalesmanProblem::TravellingSalesmanProblem(const int problem_size, const int population_count) {
    this->problem_size = problem_size;
    this->population_count = population_count;
    this->cities.reserve(problem_size);

    // TODO: make this nicer
    this->population = new int*[problem_size];
    for (int i = 0; i < problem_size; ++i) {
        this->population[i] = new int[population_count];
    }
}
