#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <chrono>
#include <array>
#include <cassert>
#include "travelling_salesman_problem.hpp"

#define POP(i,j) this->population[i * this->problem_size + j]
#define DIST(i,j) this->cities[i * this->problem_size + j]

using namespace std;

bool log_all_values = false;
bool log_best_value = true;

TravellingSalesmanProblem::TravellingSalesmanProblem(const int problem_size, float* cities,
        const int population_count, const int elite_size, const int mutation_rate) {
    this->problem_size = problem_size;
    this->population_count = population_count;
    this->elite_size = elite_size;
    this->mutation_rate = mutation_rate;
    this->fitness = vector<double>(population_count, 0.0);
    this->ranks = new int[population_count];
    this->cities = cities;
    random_device rd;
    this->gen = mt19937(rd());

    this->log_iter_freq = 100;

    // Initialize fields to be initialized later
    this->logger = nullptr;
    this->fitness_best = -1;
    this->fitness_sum = -1;

    // TODO: make this nicer
    this->population = new int[population_count * problem_size];

    // Randomly initialize the populations
    vector<int> tmp_indices(problem_size);
    for (int i = 0; i < problem_size; ++i) {
        tmp_indices[i] = i;
    }

    for (int i = 0; i < population_count; ++i) {
        shuffle(tmp_indices.begin(), tmp_indices.end(), this->gen);
        for (int j = 0; j < problem_size; ++j) {
            POP(i,j) = tmp_indices[j]; //this works
        }
    }
}

TravellingSalesmanProblem::~TravellingSalesmanProblem() {
    delete this->logger;
}

void TravellingSalesmanProblem::set_logger(Logger *_logger) {
    this->logger = _logger;
}

void TravellingSalesmanProblem::evolve(const int rank) {
    // Compute fitness
    this->rank_individuals();
    
    // Breed children
    // start = chrono::high_resolution_clock::now();
    this->breed_population();
    // stop = chrono::high_resolution_clock::now();
    // duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    // cout << "\t\tBreeding takes: " << duration.count() << "us (rank " << rank << ")" << endl;

    // Mutate population
    // start = chrono::high_resolution_clock::now();
#ifdef debug_evolve
    cout << "Before:" << endl;
    for (int i = 0; i < population_count; ++i) {
        for (int j = 0; j < problem_size; ++j) {
            cout << POP(i,j) << " ";
        }
        cout << endl;
    }
#endif

    this->mutate_population();
#ifdef debug_evolve
    cout << "After:" << endl;
    for (int i = 0; i < population_count; ++i) {
        for (int j = 0; j < problem_size; ++j) {
            cout << POP(i,j) << " ";
        }
        cout << endl;
    }
#endif
    // stop = chrono::high_resolution_clock::now();
    // duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    // cout << "\t\tMutation takes: " << duration.count() << "us (rank " << rank << ")" << endl;
}

double TravellingSalesmanProblem::solve(const int nr_epochs, const int rank) {
    this->logger->open();

#ifdef debug
    this->rank_individuals();
    for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            // cout << this->population[i+ this->population_count*j] << " ";
        }
        // cout << "\tfit: " << this->fitness[i] << endl;
    }
#endif

    for (int epoch = 0; epoch < nr_epochs; ++epoch) {
        if (epoch % this->log_iter_freq == 0) {
            cout << epoch << " of " << nr_epochs << endl;
        }
        // auto start = chrono::high_resolution_clock::now();
        this->evolve(rank);
        if (log_all_values) {
            this->logger->log_all_fitness_per_epoch(epoch, this->fitness);
        } else if (log_best_value) {
            this->logger->log_best_fitness_per_epoch(epoch, this->fitness_best);
        }

#ifdef debug
        // cout << "*** EPOCH " << epoch << " ***" << endl;
        rank_individuals();
        for (int i = 0; i < this->population_count; ++i) {
            // cout << "\tfit: " << this->fitness[i] << " rank: " << rank;
            if (this->ranks[0] == i) {
                // cout << "*";
            }
            // cout << endl;
        }
#endif
        // auto stop = chrono::high_resolution_clock::now();
        // auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        // cout << "\t" << duration.count() << " us epoch runtime (epoch " << epoch << " rank " << rank << ")" << endl;
    }

    this->rank_individuals();
    if (log_all_values) {
        this->logger->log_all_fitness_per_epoch(nr_epochs, this->fitness);
    } else if (log_best_value) {
        this->logger->log_best_fitness_per_epoch(nr_epochs, this->fitness_best);
    }

    this->logger->close();
    return this->fitness_best;
}

void TravellingSalesmanProblem::rank_individuals() {
    this->logger->LOG_WC(RANK_INDIVIDUALS_BEGIN);
    this->fitness_sum = 0.0;
    this->fitness_best = std::numeric_limits<typeof(this->fitness_best)>::max();
    for (int i = 0; i < this->population_count; ++i) {
        double new_fitness = this->evaluate_fitness(i);
        this->fitness[i] = new_fitness;
        this->fitness_sum += new_fitness;
        this->fitness_best = min(this->fitness_best, new_fitness);
    }
    iota(this->ranks, this->ranks + this->population_count, 0);
    sort(this->ranks, this->ranks + this->population_count, [this] (int i, int j) {
       return this->fitness[i] < this->fitness[j];
    });
    this->logger->LOG_WC(RANK_INDIVIDUALS_END);
}

double TravellingSalesmanProblem::evaluate_fitness(const int individual) {
    double route_distance = 0.0;
    for (int j = 0; j < this->problem_size - 1; ++j) {
        route_distance += DIST(POP(individual, j), POP(individual, j + 1));
    }
    route_distance += DIST(POP(individual, this->problem_size - 1), POP(individual, 0)); //complete the round trip
    return route_distance;
}

//this function takes up the most time in an epoch
//possible solutions:
//  * take 50% of each parent as opposed to randomly taking a sequence
//  *
void TravellingSalesmanProblem::breed(int *parent1, int *parent2, int* child) {
    //selecting gene sequences to be carried over to child
    int geneA = this->rand_range(0, this->problem_size - 1);
    int geneB = this->rand_range(0, this->problem_size - 1);
    int startGene = min(geneA, geneB);
    int endGene = max(geneA, geneB);

    set<int> selected;
    for (int i = startGene; i <= endGene; ++i) {
        child[i] = parent1[i];
        selected.insert(parent1[i]);
    }

    int index = 0;
    for (int i = 0; i < this->problem_size; ++i) {
        // If not already chosen that city
        if (selected.find(parent2[i]) == selected.end()) {
            if (index == startGene) {
                index = endGene + 1;
            }
            child[index] = parent2[i];
            index++;
        }
    }
}

void TravellingSalesmanProblem::breed_population() {
    int temp_population[this->population_count][this->problem_size];

    // Keep the best individuals
    for (int i = 0; i < this->elite_size; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            temp_population[i][j] = POP(this->ranks[i], j);
        }
    }

    vector<double> correct_fitness(this->population_count);
    for (int i = 0; i < this->population_count; ++i) {
        correct_fitness[i] = 1 / pow(this->fitness[i] / this->fitness_sum, 4);
    }

    auto dist = std::discrete_distribution<>(correct_fitness.begin(), correct_fitness.end());

    // Breed any random individuals
    for (int i = this->elite_size; i < this->population_count; ++i) {
	int rand1 = dist(gen);
	int rand2 = dist(gen);
	int* parent1;
	int* parent2;

	parent1 = this->getGene(rand1);
	parent2 = this->getGene(rand2);
        this->breed(
                parent1,
                parent2,
                temp_population[i]);

    }

    for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            POP(i, j) = temp_population[i][j];
        }
    }
}

void TravellingSalesmanProblem::mutate(int individual) {
    if (rand() % this->mutation_rate == 0) {
        int swap = rand_range(0, this->problem_size - 1);
        int swap_with = rand_range(0, this->problem_size - 1);

        int city1 = POP(individual, swap);
        int city2 = POP(individual, swap_with);
        POP(individual, swap) = city2;
        POP(individual, swap_with) = city1;
    }
}

void TravellingSalesmanProblem::mutate_population() {
    for (int i = this->elite_size / 2; i < this->population_count; ++i) {
#ifdef debug_mutate
        cout << "mutating individual:" << endl;
        for (int j = 0; j < this->problem_size; ++j) {
            cout << pop[j] << " ";
        }
        cout << endl;
#endif
        this->mutate(i);
#ifdef debug_mutate
        for (int j = 0; j < this->problem_size; ++j) {
            cout << pop[j] << " ";
        }
        cout << endl;
#endif
    }
}

int TravellingSalesmanProblem::rand_range(const int &a, const int&b) {
    return (rand() % (b - a + 1) + a);
}

int* TravellingSalesmanProblem::getRanks() {
    return (this->ranks);
}

double TravellingSalesmanProblem::getFitness(int indivIdx) {
    return (this->fitness)[indivIdx];
}

void TravellingSalesmanProblem::setFitness(int indivIdx, double fitness) {
    (this->fitness)[indivIdx] = fitness;
}

double TravellingSalesmanProblem::getMinFitness() {
    return *min_element((this->fitness).begin(), (this->fitness).end());
}

int* TravellingSalesmanProblem::getGene(int indivIdx) {
    int* pop = new int[this->problem_size];
    for (int j = 0; j < this->problem_size; ++j){
	    pop[j] = POP(indivIdx, j);
    } 
    return pop;
}

