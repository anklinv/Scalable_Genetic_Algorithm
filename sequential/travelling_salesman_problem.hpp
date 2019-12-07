#ifndef DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP
#define DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP

#include <random>
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
    /// \param cities matrix of the city distances
    /// \param population_count size of the population
    /// \param elite_size number of individuals that survive for sure
    /// \param mutation_rate 1/mutation_rate is the probability that an individual gets a mutation
    TravellingSalesmanProblem(int problem_size, float* cities, int population_count, int elite_size, int mutation_rate,
            int verbose, int log_freq);
    ~TravellingSalesmanProblem();

    /// For debug printing
    int verbose;

    /// Number of nodes in the graph
    int problem_size;

    /// Number of individuals in the population
    int population_count;

    /// Number of individuals that survive every iteration
    int elite_size;

    /// Probability of each index to randomly switch with a random second index
    int mutation_rate;

    /// 2D coordinates of the cities
    float* cities;

    void set_logger(Logger *_logger);

    /// Solve the problem by evolving for a given number of steps.
    /// \param nr_epochs number of steps to evolve
    /// \return the best length of the best path found by the algorithm
    double solve(int nr_epochs, int rank);
    
    
    /// Getter method to access the ranks after execution of the algorithm. For Island.
    int* getRanks();
    
    /// Getter method to access the genes of the individuals of the population. For Island.
    Int* getGenes();
    
    /// Getter method to access the fitness of a single individual. For Island.
    double getFitness(int indivIdx);
    
    /// Setter method to set the fitness of a single individual. For Island.
    void setFitness(int indivIdx, double newFitness);
    
    /// Returns the minimum (best) fitness value. For Island.
    double getMinFitness();
    
    
    /// Getter method to access the "gene" of a single individual (??)
    int* getGene(int indivIdx);
    

private:
    /// Logger object
    Logger *logger;

    /// Pointer to the population indices, which has size population_count * problem_size
    Int* population;

    /// Fitness of individuals. i-th element is the path length of i-th individual
    vector<double> fitness;
    double fitness_sum;
    double fitness_best;

    /// Sorted ranks of individuals. i-th element is the index of the i-th best individual
    int* ranks;

    /// For randomness
    std::mt19937 gen;

    /// How often to write
    int log_iter_freq;
    
    /// A counter to count the number of evolution steps done since object creation. This is necessary because
    /// an Island calls solve() multiple times.
    int evolutionCounter;

    /// Calculate the fitness of an individual, which is the length of the closed path in the graph and return it.
    /// \param index of individual
    /// \return the length of the path in the graph
    double evaluate_fitness(int individual);

    /// Run a single iteration of selection, breeding and mutation
    void evolve(int rank);

    /// Calculate the fitness of all individuals, save it in this->fitness and also calculate the ranks and save those
    /// in this->ranks
    void rank_individuals();

    /// Breed two individuals into a new one. Take a random sequence of parent1 and merge the remaining path from parent2
    /// while making sure that there is nothing that repeats.
    /// \param parent1 index of mother of the child
    /// \param parent2 index of father of the child
    /// \param child mix of mother and father
    void breed(int parent1, int parent2, Int* child);

    /// Apply breeding to the whole population by taking random individuals and breeding them. Make sure that all elite
    /// members stay unchanged.
    void breed_population();

    /// Mutate a single individual by randomly flipping some cities in the path
    /// \param index to an individual
    void mutate(int individual);

    /// Apply mutation to the whole population
    void mutate_population();

    int rand_range(const int &a, const int&b);
};


#endif //DPHPC_PROJECT_TRAVELLING_SALESMAN_PROBLEM_HPP
