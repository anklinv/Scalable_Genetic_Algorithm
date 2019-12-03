#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <chrono>
#include <array>
#include <cassert>
#include <immintrin.h> // for SIMD
#include <fstream> // for writing CSV files
#include "travelling_salesman_problem.hpp"

#define SAFE_DEBUG
#ifdef SAFE_DEBUG
#define VAL_POP(i,j) assert(i >= 0); assert(i < this->population_count); assert(j >= 0); assert(j < this->problem_size)
#define VAL_DIST(i,j) assert(i >= 0); assert(i < this->problem_size); assert(j >= 0); assert(j < this->problem_size)
#else
#define VAL_POP(i,j)
#define VAL_DIST(i,j)
#endif

#define POP(i,j) this->population[i * this->problem_size + j]
#define DIST(i,j) this->cities[i * this->problem_size + j]

using namespace std;

bool log_all_values = false;
bool log_best_value = true;


/*
 For microbenchmarking.
 */

typedef std::vector<int> vi;

typedef std::chrono::high_resolution_clock hrClock;
typedef std::chrono::time_point<hrClock> hrTime;

typedef std::chrono::nanoseconds hrNanos;

vi rndRuntimes;
vi chunkRuntimes;
vi splitRuntimes;

hrClock myClock;

/*
 For debugging.
 */

#define assertm(exp, msg) assert(((void)msg, exp))


TravellingSalesmanProblem::TravellingSalesmanProblem(const int problem_size, float* cities,
        const int population_count, const int elite_size, const int mutation_rate, const int verbose) {
    this->verbose = verbose;
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
    this->population = new Int[population_count * problem_size];
    
    this->evolutionCounter = 0;

    // Randomly initialize the populations
    vector<Int> tmp_indices(problem_size);
    for (Int i = 0; i < problem_size; ++i) {
        tmp_indices[i] = i;
    }

    for (int i = 0; i < population_count; ++i) {
        shuffle(tmp_indices.begin(), tmp_indices.end(), this->gen);
        for (int j = 0; j < problem_size; ++j) {
            VAL_POP(i, j);
            POP(i,j) = tmp_indices[j]; //this works
        }
    }
    
#ifdef microbenchmark_breed
    cout << "Size of one gene index is " << sizeof(Int) << " bytes " << endl;
    cout << "Size of entire gene is " << sizeof(Int) * problem_size << " bytes " << endl;
    cout << "Size of entire population is (w/o fitness, temp_population, distances) ";
    cout << sizeof(Int) * problem_size * population_count << " bytes " << endl;
#endif
    
}

TravellingSalesmanProblem::~TravellingSalesmanProblem() {
    
#ifdef microbenchmark_breed
    double numCrossovers = this->evolutionCounter * (this->population_count - this->elite_size);
    
    cout << "num crossovers was " << numCrossovers << endl;
    
    double accRnd = (double)accumulate(rndRuntimes.begin(), rndRuntimes.end(), 0) / (double)1e9;
    double accChunk = (double)accumulate(chunkRuntimes.begin(), chunkRuntimes.end(), 0) / (double)1e9;
    double accSplit = (double)accumulate(splitRuntimes.begin(), splitRuntimes.end(), 0) / (double)1e9;
    
    cout << "mean runtime rnd (ns): (total runtime " << accRnd << "s)" << endl;
    cout << (double)accumulate(rndRuntimes.begin(), rndRuntimes.end(), 0)/numCrossovers << endl;
    cout << "mean runtime chunk (ns): (total runtime " << accChunk << "s)" << endl;
    cout << (double)accumulate(chunkRuntimes.begin(), chunkRuntimes.end(), 0)/numCrossovers << endl;
    cout << "mean runtime split (ns): (total runtime " << accSplit << "s)" << endl;
    cout << (double)accumulate(splitRuntimes.begin(), splitRuntimes.end(), 0)/numCrossovers << endl;
    
    string rndRuntimesFile = "rndRuntimesFile.csv";
    string chunkRuntimesFile = "chunkRuntimesFile.csv";
    string splitRuntimesFile = "splitRuntimesFile.csv";
    
    std::ofstream outputFileStream;
    
    outputFileStream.open(rndRuntimesFile);
    for(int idx = 0; idx < rndRuntimes.size(); idx++) {
        outputFileStream << rndRuntimes[idx];
        outputFileStream << "\n";
    }
    outputFileStream.close();
    
    outputFileStream.open(chunkRuntimesFile);
    for(int idx = 0; idx < chunkRuntimes.size(); idx++) {
        outputFileStream << chunkRuntimes[idx];
        outputFileStream << "\n";
    }
    outputFileStream.close();
    
    outputFileStream.open(splitRuntimesFile);
    for(int idx = 0; idx < splitRuntimes.size(); idx++) {
        outputFileStream << splitRuntimes[idx];
        outputFileStream << "\n";
    }
    outputFileStream.close();
#endif
    
    this->logger->close();
    delete this->logger;
}

void TravellingSalesmanProblem::set_logger(Logger *_logger) {
    this->logger = _logger;
    this->logger->open();
    
    // necessary for island. this way the rank vector can be used after object
    // creation. can't be called in the constructor as the logger is used inside
    // rank_individuals().
    this->rank_individuals();
}

void TravellingSalesmanProblem::evolve(const int rank) {
    // Compute fitness
    this->rank_individuals();
    
    // Breed children
    this->breed_population();

    // Mutate population
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
    
}

double TravellingSalesmanProblem::solve(const int nr_epochs, const int rank) {

/*#ifdef debug
    this->rank_individuals();
    for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            // cout << this->population[i+ this->population_count*j] << " ";
        }
        // cout << "\tfit: " << this->fitness[i] << endl;
    }
#endif*/

    for (int epoch = 0; epoch < nr_epochs; ++epoch) {
        this->logger->LOG_WC(EPOCH_BEGIN);
        if (this->verbose > 0) {
            if (epoch % this->log_iter_freq == 0) {
                cout << epoch << " of " << nr_epochs << endl;
            }
        }
        this->evolve(rank);
        
        // - if the best fitness is logged, the best fitness before the previous evolution step is logged (due to ranking)
        // - the best fitness after the very last evolution is not logged
        if (log_all_values) {
            this->logger->log_all_fitness_per_epoch(this->evolutionCounter, this->fitness);
        } else if (log_best_value) {
            this->logger->log_best_fitness_per_epoch(this->evolutionCounter, this->fitness_best);
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
        this->evolutionCounter = this->evolutionCounter + 1;
        this->logger->LOG(BEST_FITNESS, this->fitness_best);
        this->logger->LOG_WC(EPOCH_END);
    }
    
    // Island assumes the ranks to be sorted before a migration starts
    this->rank_individuals();

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
        VAL_POP(individual, j);
        VAL_POP(individual, j+1);
        VAL_DIST(POP(individual, j), POP(individual, j + 1));
        route_distance += DIST(POP(individual, j), POP(individual, j + 1));
    }
    VAL_POP(individual, this->problem_size - 1);
    VAL_POP(individual, 0);
    VAL_DIST(POP(individual, this->problem_size - 1), POP(individual, 0));
    route_distance += DIST(POP(individual, this->problem_size - 1), POP(individual, 0)); //complete the round trip
    return route_distance;
}

//this function takes up the most time in an epoch
//possible solutions:
//  * take 50% of each parent as opposed to randomly taking a sequence
//  *
void TravellingSalesmanProblem::breed(const int parent1, const int parent2, Int* child) {
        
    bool useSet = false;
    
    if(useSet == false) {

#ifdef microbenchmark_breed
        hrTime tStart, tEnd;
        int delta;
        
        tStart = myClock.now();
#endif
        
        //selecting gene sequences to be carried over to child
        int geneA = this->rand_range(0, this->problem_size - 1);
        int geneB = this->rand_range(0, this->problem_size - 1);
        int startGene = min(geneA, geneB);
        int endGene = max(geneA, geneB);
        
#ifdef microbenchmark_breed
        tEnd = myClock.now();
        delta = std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
        rndRuntimes.push_back(delta);
        
        tStart = myClock.now();
#endif
        
        /*
         Experiment with mask.
         */
        
        Int mask[problem_size]; // size of a single individual
        for(int idx = 0; idx < problem_size; idx++) {
            mask[idx] = 1;
        }
        // cities are indexed 0, ..., (problem_size-1)
        
            
        for (int i = startGene; i <= endGene; ++i) {
            child[i] = POP(parent1, i);
            //assertm(0 <= POP(parent1, i) && POP(parent1, i) <= problem_size-1, "segfault mask, loop chunk");
            //assertm(mask[POP(parent1, i)] == 1, "gene part is already masked, loop chunk");
            mask[POP(parent1, i)] = 0; // no comparisons
        }
        
        /*int sum = 0;
        for(int i = 0; i < problem_size; i++) {
            sum += mask[i];
        }
        assertm(sum == problem_size - ((endGene-startGene)+1), "num elements masked differs from num elements in chunk");*/
        
#ifdef microbenchmark_breed
        tEnd = myClock.now();
        delta = std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
        chunkRuntimes.push_back(delta);

        tStart = myClock.now();
#endif
        
        int parent2idx = 0;
        
        for(int childIdx = 0; childIdx < startGene;) {
            child[childIdx] = POP(parent2, parent2idx);
            childIdx = childIdx + (1 & mask[POP(parent2, parent2idx)]);
            parent2idx++;
        }
        
        for(int childIdx = endGene+1; childIdx < problem_size;) {
            child[childIdx] = POP(parent2, parent2idx);
            childIdx = childIdx + (1 & mask[POP(parent2, parent2idx)]);
            parent2idx++;
        }
            
            
        /*bool sthswrong = false;
        
        for(int i = 0; i < problem_size; i++) {
            bool contained = false;
            for(int j = 0; j < problem_size; j++) {
                if(child[j] == i) {
                    contained = true;
                }
            }
            if(!contained) {
                //assertm(false, "error, child misses at least one gene element");
                sthswrong = true;
            }
        }
        
        if(sthswrong) {
            cout << "-----------------------" << endl;
            for(int i = 0; i < problem_size; i++) {
                cout << child[i] << " ";
            } cout << endl;
            cout << "-----------------------" << endl;
            exit(1);
        }*/
        
        
#ifdef microbenchmark_breed
        tEnd = myClock.now();
        delta = std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
        splitRuntimes.push_back(delta);
#endif
        
    } else { // useSet == true;
            
#ifdef microbenchmark_breed
        hrTime tStart, tEnd;
        int delta;
        
        tStart = myClock.now();
#endif
                    
        //selecting gene sequences to be carried over to child
        int geneA = this->rand_range(0, this->problem_size - 1);
        int geneB = this->rand_range(0, this->problem_size - 1);
        int startGene = min(geneA, geneB);
        int endGene = max(geneA, geneB);
                    
#ifdef microbenchmark_breed
        tEnd = myClock.now();
        delta = std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
        rndRuntimes.push_back(delta);
        
        tStart = myClock.now();
#endif
            
        set<Int> selected;
        
        for (int i = startGene; i <= endGene; ++i) {
            VAL_POP(parent1, i);
            child[i] = POP(parent1, i);
            selected.insert(POP(parent1, i));
        }
        
#ifdef microbenchmark_breed
        tEnd = myClock.now();
        delta = std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
        chunkRuntimes.push_back(delta);

        tStart = myClock.now();
#endif
        
        int index = 0;
        for (int i = 0; i < this->problem_size; ++i) {
            // If not already chosen that city
            VAL_POP(parent2, i);
            if (selected.find(POP(parent2, i)) == selected.end()) {
                if (index == startGene) {
                    index = endGene + 1;
                }
                child[index] = POP(parent2, i);
                index++;
            }
        }
            
#ifdef microbenchmark_breed
        tEnd = myClock.now();
        delta = std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
        splitRuntimes.push_back(delta);
#endif
        
    } // end if else
    
}

void TravellingSalesmanProblem::breed_population() {
    this->logger->LOG_WC(BREED_POPULATION_BEGIN);
    
    Int temp_population[this->population_count][this->problem_size];

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
        this->breed(rand1, rand2, temp_population[i]);
    }

    for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            VAL_POP(i, j);
            POP(i, j) = temp_population[i][j];
        }
    }
    this->logger->LOG_WC(BREED_POPULATION_END);
}

void TravellingSalesmanProblem::mutate(const int individual) {
    if (rand() % this->mutation_rate == 0) {
        int swap = rand_range(0, this->problem_size - 1);
        int swap_with = rand_range(0, this->problem_size - 1);

        VAL_POP(individual, swap);
        VAL_POP(individual, swap_with);
        int city1 = POP(individual, swap);
        int city2 = POP(individual, swap_with);
        POP(individual, swap) = city2;
        POP(individual, swap_with) = city1;
    }
}

void TravellingSalesmanProblem::mutate_population() {
    this->logger->LOG_WC(MUTATE_POPULATION_BEGIN);
    
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
    
    this->logger->LOG_WC(MUTATE_POPULATION_END);
}

int TravellingSalesmanProblem::rand_range(const int &a, const int&b) {
    return (rand() % (b - a + 1) + a);
}

int* TravellingSalesmanProblem::getRanks() { // for Island
    return ranks;
}

Int* TravellingSalesmanProblem::getGenes() { // for Island
    return population;
}

double TravellingSalesmanProblem::getFitness(int indivIdx) { // for Island
    return fitness[indivIdx];
}

void TravellingSalesmanProblem::setFitness(int indivIdx, double newFitness) { // for Island
    fitness[indivIdx] = newFitness;
}

double TravellingSalesmanProblem::getMinFitness() { // for Island
    return *min_element(fitness.begin(), fitness.end());
}
