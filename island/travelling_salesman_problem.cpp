
#include "travelling_salesman_problem.hpp"

using namespace std;

TravellingSalesmanProblem::TravellingSalesmanProblem(const int problem_size, const int population_count,
        const int elite_size, const int mutation_rate) {
    
    this->problem_size = problem_size; // number of nodes of the TSP graph
    this->population_count = population_count;
    this->elite_size = elite_size;
    this->mutation_rate = mutation_rate;
    this->fitness = vector<double>(population_count, 0.0);
    this->ranks = new int[population_count];
    this->cities = new int[problem_size * problem_size];
    this->gen = mt19937(this->rd());

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

TravellingSalesmanProblem::~TravellingSalesmanProblem() {
    if (this->logger) delete this->logger;
}

void TravellingSalesmanProblem::set_logger(Logger *logger) {
    this->logger = logger;
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
    this->logger->open();

#ifdef debug
    this->rank_individuals();
    for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            cout << this->population[i][j] << " ";
        }
        cout << "\tfit: " << this->fitness[i] << endl;
    }
#endif

    for (int epoch = 0; epoch < nr_epochs; ++epoch) {
        this->evolve();
        this->logger->log_best_fitness_per_epoch(epoch, this->fitness);
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
    this->logger->log_best_fitness_per_epoch(nr_epochs, this->fitness);

    this->logger->close();
    
    return this->fitness_best;
}

void TravellingSalesmanProblem::rank_individuals() {
    this->fitness_sum = 0.0;
    this->fitness_best = std::numeric_limits<typeof(this->fitness_best)>::max();

    for (int i = 0; i < this->population_count; ++i) {
        double fitness = this->evaluate_fitness(this->population[i]);
        this->fitness[i] = fitness;
        this->fitness_sum += fitness;
        this->fitness_best = min(this->fitness_best, fitness);
    }
    iota(this->ranks, this->ranks + population_count, 0);
    sort(this->ranks, this->ranks + population_count, [this] (int i, int j) {
       return this->fitness[i] < this->fitness[j];
    });
}

double TravellingSalesmanProblem::evaluate_fitness(const int *individual) {
    double route_distance = 0.0;
    for (int i = 0; i < this->problem_size - 1; ++i) {
        route_distance += this->cities[individual[i] + problem_size * individual[i+1]];		//matrix lookup for a distance between two cities
    }
    route_distance += this->cities[individual[problem_size-1] + problem_size * individual[0]];	//complete the round trip
    return route_distance;
}

void TravellingSalesmanProblem::breed(int *parent1, int *parent2, int* child) {
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
            temp_population[i][j] = this->population[this->ranks[i]][j];
        }
    }

    vector<double> correct_fitness;
    correct_fitness.reserve(this->population_count);
    for (auto f : this->fitness) {
        correct_fitness.push_back(1 / pow(f / this->fitness_sum, 4));
    }

    //auto dist = std::uniform_int_distribution(0, population_count - 1);
    auto dist = std::discrete_distribution(correct_fitness.begin(), correct_fitness.end());

    /*
    int fittest_n = 6;
    vector<int> pop(population_count);
    iota(pop.begin(), pop.end(), 0);
     */

    // Breed any random individuals
    for (int i = this->elite_size; i < population_count; ++i) {

        /*
        vector<int> tournament1, tournament2;
        tournament1.reserve(fittest_n);
        tournament2.reserve(fittest_n);
        std::sample(pop.begin(), pop.end(), back_inserter(tournament1), fittest_n, gen);
        std::sample(pop.begin(), pop.end(), back_inserter(tournament2), fittest_n, gen);
        int best_t1;
        double best_t1_fitness = INFINITY;
        for (auto t1 : tournament1) {
            if (this->fitness[t1] < best_t1_fitness) {
                best_t1 = t1;
                best_t1_fitness = this->fitness[t1];
            }
        }
        int best_t2;
        double best_t2_fitness = INFINITY;
        for (auto t2 : tournament2) {
            if (this->fitness[t2] < best_t2_fitness) {
                best_t2 = t2;
                best_t2_fitness = this->fitness[t2];
            }
        }
        this->breed(
                this->population[best_t1],
                this->population[best_t2],
                temp_population[i]);
        */

        this->breed(
                this->population[dist(gen)],
                this->population[dist(gen)],
                temp_population[i]);

    }

    for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            this->population[i][j] = temp_population[i][j];
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
    for (int i = this->elite_size / 2; i < population_count; ++i) {
        this->mutate(population[i]);
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

double TravellingSalesmanProblem::getMaxFitness() {
    return *max_element((this->fitness).begin(), (this->fitness).end());
}

int* TravellingSalesmanProblem::getGene(int indivIdx) {
    return (this->population)[indivIdx];
}


