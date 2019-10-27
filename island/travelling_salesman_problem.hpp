#ifndef DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP
#define DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "../logging/logging.hpp"

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
    /// \param mutation_rate 1/mutation_rate is the probability that an individual gets a mutation
    TravellingSalesmanProblem(int problem_size, int population_count, int elite_size, int mutation_rate);
    
    /// Copy constructor. Only copies the general settings. The population is initialized randomly.
    /// \param tsp a TravellingSalesmanProblem
    TravellingSalesmanProblem(const TravellingSalesmanProblem& tsp);
    
    ~TravellingSalesmanProblem();

    /// Number of nodes in the graph
    int problem_size;

    /// Number of individuals in the population
    int population_count;

    /// Number of individuals that survive every iteration
    int elite_size;

    /// Probability of each index to randomly switch with a random second index
    int mutation_rate;

    /// 2D coordinates of the cities
    int* cities;

    void set_logger(Logger *logger);

    /// Solve the problem by evolving for a given number of steps.
    /// \param nr_epochs number of steps to evolve
    /// \return the best length of the best path found by the algorithm
    double solve(int nr_epochs);
    
    /// Getter method to access the ranks after the execution of the algorithm (for Island)
    int* getRanks();
    
    /// Getter method to access the fitness of a single individual (for Island)
    double getFitness(int indivIdx);
    
    /// Setter method to set the fitness of a single individual (for Island)
    void setFitness(int indivIdx, double fitness);
    
    /// Getter method to access the "gene" of a single individual (for Island)
    int* getGene(int indivIdx);
    
    /// Returns the maximum value stored inside the vector fitness (for Island)
    double getMinFitness();
    

private:
    /// Logger object
    Logger *logger;

    /// Pointer to the population indices, which has size population_count * problem_size
    int** population;

    /// Fitness of individuals. i-th element is the path length of i-th individual
    vector<double> fitness;
    double fitness_sum;
    double fitness_best;

    /// Sorted ranks of individuals. i-th element is the index of the i-th best individual
    int* ranks;

    /// For randomness
    std::mt19937 gen;

    /// Calculate the fitness of an individual, which is the length of the closed path in the graph and return it.
    /// \param individual pointer to an array of size problem_size
    /// \return the length of the path in the graph
    double evaluate_fitness(const int* individual);

    /// Run a single iteration of selection, breeding and mutation
    void evolve();

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

    int rand_range(const int &a, const int&b);
};


#endif //DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP
