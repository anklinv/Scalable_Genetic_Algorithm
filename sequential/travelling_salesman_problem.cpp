#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <chrono>
#include "travelling_salesman_problem.hpp"

using namespace std;

TravellingSalesmanProblem::TravellingSalesmanProblem(const int problem_size, const int population_count,
        const int elite_size, const int mutation_rate) {
    this->problem_size = problem_size;
    this->population_count = population_count;
    this->elite_size = elite_size;
    this->mutation_rate = mutation_rate;
    this->fitness = vector<double>(population_count, 0.0);
    this->ranks = new int[population_count];
    this->cities = new int[problem_size * problem_size];
    random_device rd;
    this->gen = mt19937(rd());

    // TODO: make this nicer
    this->population = new int[population_count*problem_size];

    // Randomly initialize the populations
    vector<int> tmp_indices(problem_size);
    for (int i = 0; i < problem_size; ++i) {
        tmp_indices[i] = i;
    }

    for (int i = 0; i < population_count; ++i) {
        shuffle(tmp_indices.begin(), tmp_indices.end(), this->gen);
        for (int j = 0; j < problem_size; ++j) {
            this->population[i + population_count * j] = tmp_indices[j]; //this works
        }
    }
}

TravellingSalesmanProblem::~TravellingSalesmanProblem() {
    if (this->logger) delete this->logger;
}

void TravellingSalesmanProblem::set_logger(Logger *logger) {
    this->logger = logger;
}

void TravellingSalesmanProblem::evolve(const int rank, const int epoch) {
    // Compute fitness
    auto start = chrono::high_resolution_clock::now();
    this->rank_individuals();
    auto stop = chrono::high_resolution_clock::now();
    auto rank_duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    
    // Breed children
    start = chrono::high_resolution_clock::now();
    this->breed_population();
    stop = chrono::high_resolution_clock::now();
    auto breed_duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    // Mutate population
    start = chrono::high_resolution_clock::now();
    this->mutate_population();
    stop = chrono::high_resolution_clock::now();
    auto mutate_duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    
    this->logger->log_timing_per_epoch(epoch, rank_duration.count(), breed_duration.count(), mutate_duration.count());
}

double TravellingSalesmanProblem::solve(const int nr_epochs, const int rank) {
    
    this->logger->open();
    this->logger->open_timing();

#ifdef debug
    this->rank_individuals();
    for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            cout << this->population[i+ this->population_count*j] << " ";
        }
        cout << "\tfit: " << this->fitness[i] << endl;
    }
#endif

    for (int epoch = 0; epoch < nr_epochs; ++epoch) {
        this->evolve(rank, epoch);
        this->logger->log_best_fitness_per_epoch(epoch, this->fitness);
#ifdef debug
        cout << "*** EPOCH " << epoch << " ***" << endl;
        rank_individuals();
        for (int i = 0; i < this->population_count; ++i) {
            cout << "\tfit: " << this->fitness[i] << " rank: " << rank;
            if (this->ranks[0] == i) {
                cout << "*";
            }
            cout << endl;
        }
#endif
    }

    this->rank_individuals();
    this->logger->log_best_fitness_per_epoch(nr_epochs, this->fitness);

    this->logger->close();
    this->logger->close_timing();
    return this->fitness_best;
}

void TravellingSalesmanProblem::rank_individuals() {
    this->fitness_sum = 0.0;
    this->fitness_best = std::numeric_limits<typeof(this->fitness_best)>::max();
    int* pop = new int[this->problem_size];
    for (int i = 0; i < this->population_count; ++i) {
	pop = this->getGene(i);
        double fitness = this->evaluate_fitness(pop);
        this->fitness[i] = fitness;
        this->fitness_sum += fitness;
        this->fitness_best = min(this->fitness_best, fitness);
    }
    iota(this->ranks, this->ranks + this->population_count, 0);
    sort(this->ranks, this->ranks + this->population_count, [this] (int i, int j) {
       return this->fitness[i] < this->fitness[j];
    });
}

double TravellingSalesmanProblem::evaluate_fitness(const int *individual) {
    double route_distance = 0.0;
    for (int i = 0; i < this->problem_size - 1; ++i) {
        route_distance += this->cities[individual[i] + this->problem_size * individual[i+1]];		//matrix lookup for a distance between two cities
    }
    route_distance += this->cities[individual[this->problem_size-1] + this->problem_size * individual[0]];	//complete the round trip
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
            temp_population[i][j] = this->population[this->ranks[i] + this->population_count * j]; //this works
        }
    }

    vector<double> correct_fitness;
    correct_fitness.reserve(this->population_count);
    for (auto f : this->fitness) {
        correct_fitness.push_back(1 / pow(f / this->fitness_sum, 4));
    }

    auto dist = std::discrete_distribution(correct_fitness.begin(), correct_fitness.end());

    // Breed any random individuals
    for (int i = this->elite_size; i < this->population_count; ++i) {
	int rand1 = dist(gen);
	int rand2 = dist(gen);
	int* parent1 = new int[this->problem_size];
	int* parent2 = new int[this->problem_size];
	while (rand1 == rand2) {
		rand2 = dist(gen);
	}
	parent1 = this->getGene(rand1);
	parent2 = this->getGene(rand2);
        this->breed(
                parent1,
                parent2,
                temp_population[i]);

    }

    for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            this->population[i + this->population_count * j] = temp_population[i][j];
        }
    }
}

void TravellingSalesmanProblem::mutate(int *individual) {
    if (rand() % this->mutation_rate == 0) {
        int swap = rand_range(0, this->problem_size - 1);
        int swap_with = rand_range(0, this->problem_size - 1);
        int city1 = individual[swap];
        int city2 = individual[swap_with];
        individual[swap] = city2;
        individual[swap_with] = city1;
    }
}

void TravellingSalesmanProblem::mutate_population() {
    int* pop = new int[this->problem_size];
    for (int i = this->elite_size / 2; i < this->population_count; ++i) {
	pop = this->getGene(i);
        this->mutate(pop);
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
	pop[j] = this->population[indivIdx + this->population_count * j];
    } 
    return pop;
}

