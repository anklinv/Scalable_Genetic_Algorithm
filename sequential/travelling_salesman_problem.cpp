#include <random>
#include <iostream>
#include <algorithm>
#include <set>
#include "travelling_salesman_problem.hpp"

using namespace std;

TravellingSalesmanProblem::TravellingSalesmanProblem(const int problem_size, const int population_count,
        const int elite_size, const double mutation_rate) {
    this->problem_size = problem_size;
    this->population_count = population_count;
    this->elite_size = elite_size;
    this->mutation_rate = mutation_rate;
    this->fitness = vector<double>(population_count, 0.0);
    this->ranks = vector<int>(population_count);
    this->cities.reserve(problem_size);
    this->gen = mt19937(this->rd());
    this->nr_mutations = std::uniform_int_distribution<>(0, this->population_count * this->mutation_rate);
    this->random_gene = std::uniform_int_distribution(0, this->problem_size - 1);
    this->random_individual = std::uniform_int_distribution(0, this->population_count - 1);

    // TODO: make this nicer
    this->population = new int *[population_count];
    for (int i = 0; i < population_count; ++i) {
        this->population[i] = new int[problem_size];
    }

    // Randomly initialize the populations
    vector<int> tmp_indices(problem_size);
    for (int i = 0; i < problem_size; ++i) {
        tmp_indices[i] = i;
    }

    for (int i = 0; i < population_count; ++i) {
        shuffle(tmp_indices.begin(), tmp_indices.end(), this->gen);
        for (int j = 0; j < problem_size; ++j) {
            this->population[i][j] = tmp_indices[j];
        }
    }
}

void TravellingSalesmanProblem::evolve() {
    // Compute fitness
    this->rank_individuals();

    // Breed children
    this->breed_population();

    // Mutate population
    this->mutate_population();
}


double TravellingSalesmanProblem::solve(const int nr_epochs) {
    this->rank_individuals();
    double initial_distance = *min_element(this->fitness.begin(), this->fitness.end());

#ifdef debug
    for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            cout << this->population[i][j] << " ";
        }
        cout << "\tfit: " << this->fitness[i] << endl;
    }
#endif

    for (int epoch = 0; epoch < nr_epochs; ++epoch) {
        this->evolve();
#ifdef debug
        cout << "*** EPOCH " << epoch << " ***" << endl;
        rank_individuals();
        for (int i = 0; i < this->population_count; ++i) {
            for (int j = 0; j < this->problem_size; ++j) {
                cout << this->population[i][j] << " ";
            }
            cout << "\tfit: " << this->fitness[i];
            if (this->ranks[0] == i) {
                cout << "*";
            }
            cout << endl;
        }
#endif
    }

    this->rank_individuals();
    double final_distance = *min_element(this->fitness.begin(), this->fitness.end());
    
    return final_distance;
}

void TravellingSalesmanProblem::rank_individuals() {
    for (int i = 0; i < this->population_count; ++i) {
        this->fitness[i] = this->evaluate_fitness(this->population[i]);
    }
    iota(this->ranks.begin(), this->ranks.end(), 0);
    sort(this->ranks.begin(), this->ranks.end(), [this] (int i, int j) {
       return this->fitness[i] < this->fitness[j];
    });
}

double TravellingSalesmanProblem::evaluate_fitness(int *individual) {
    double route_distance = 0.0;
    for (int i = 0; i < this->problem_size - 1; ++i) {
        route_distance += distance(this->cities[individual[i]], this->cities[individual[i+1]]);
    }
    route_distance += distance(this->cities[individual[problem_size-1]], this->cities[individual[0]]);
    return route_distance;
}

double TravellingSalesmanProblem::distance(City &a, City &b) {
    return sqrt(pow(a.x - b.x, 2.0) + pow(a.y - b.y, 2.0));
}

void TravellingSalesmanProblem::breed(int *parent1, int *parent2, int* child) {
    int geneA = this->random_gene(gen);
    int geneB = this->random_gene(gen);
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
            temp_population[i][j] = this->population[this->ranks[i]][j];
        }
    }

    // Breed any random individuals
    for (int i = this->elite_size; i < population_count; ++i) {
        this->breed(
                this->population[random_individual(gen)],
                this->population[random_individual(gen)],
                temp_population[i]);
    }

    for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            this->population[i][j] = temp_population[i][j];
        }
    }
}

void TravellingSalesmanProblem::mutate(int *individual) {
    int changes = nr_mutations(gen);
    vector<int> tmp(this->population_count);
    vector<int> swap, swap_with;
    swap.reserve(changes);
    swap_with.reserve(changes);
    sample(tmp.begin(), tmp.end(), back_inserter(swap), changes, gen);
    sample(tmp.begin(), tmp.end(), back_inserter(swap_with), changes, gen);
    for (int i = 0; i < changes; ++i) {
        int city1 = individual[swap[i]];
        int city2 = individual[swap_with[i]];
        individual[swap[i]] = city2;
        individual[swap_with[i]] = city1;
    }
}

void TravellingSalesmanProblem::mutate_population() {
    for (int i = 0; i < population_count; ++i) {
        this->mutate(population[i]);
    }
}
