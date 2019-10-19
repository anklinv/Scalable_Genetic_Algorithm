#include <vector>

#ifndef DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP
#define DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP

using namespace std;

struct City {
    double x, y;
};

class TravellingSalesmanProblem {
public:
    /// Generate a Problem object that can be solved afterwards with .solve()
    /// \param problem_size number of nodes in the graph
    /// \param population_count size of the population
    /// \param elite_size number of individuals that survive for sure
    /// \param mutation_rate rate of mutations
    TravellingSalesmanProblem(int problem_size, int population_count, int elite_size, double mutation_rate);

    /// Number of nodes in the graph
    int problem_size;

    /// Number of individuals in the population
    int population_count;

    /// Number of individuals that survive every iteration
    int elite_size;

    /// Probability of each index to randomly switch with a random second index
    double mutation_rate;

    /// 2D coordinates of the cities
    vector<City> cities;

    /// Pointer to the population indices, which has size population_count * problem_size
    int** population;

    /// Fitness of individuals. i-th element is the path length of i-th individual
    vector<double> fitness;

    /// Sorted ranks of individuals. i-th element is the index of the i-th best individual
    vector<int> ranks;

    /// For randomness
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> nr_mutations;
    std::uniform_int_distribution<> random_gene;
    std::uniform_int_distribution<> random_individual;

    /// Calculate the fitness of an individual, which is the length of the closed path in the graph and return it.
    /// \param individual pointer to an array of size problem_size
    /// \return the length of the path in the graph
    double evaluate_fitness(int* individual);

    /// Run a single iteration of selection, breeding and mutation
    void evolve();

    /// Solve the problem by evolving for a given number of steps.
    /// \param nr_epochs number of steps to evolve
    /// \return the best length of the best path found by the algorithm
    double solve(int nr_epochs);

    /// Calculate the fitness of all individuals, save it in this->fitness and also calculate the ranks and save those
    /// in this->ranks
    void rank_individuals();

    /// Breed two individuals into a new one. Take a random sequence of parent1 and merge the remaining path from parent2
    /// while making sure that there is nothing that repeats.
    /// \param parent1 mother of the child
    /// \param parent2 father of the child
    /// \param child mix of mother and father
    void breed(int *parent1, int *parent2, int* child);

    /// Apply breeding to the whole population by taking random individuals and breeding them. Make sure that all elite
    /// members stay unchanged.
    void breed_population();

    /// Mutate a single individual by randomly flipping some cities in the path
    /// \param individual an array of size this->problem_size
    void mutate(int *individual);

    /// Apply mutation to the whole population
    void mutate_population();

    /// Calculates the distance between two cities
    /// \param a origin city
    /// \param b destination city
    /// \return euclidean distance from a to b
    static double distance(City &a, City &b);
};


#endif //DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP
