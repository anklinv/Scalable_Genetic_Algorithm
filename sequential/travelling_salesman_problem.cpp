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
#include <stdlib.h> // for aligned alloc
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

vi rndRuntimes;
vi chunkRuntimes;
vi splitRuntimes;

hrClock myClock;

vi sumAndMinRuntimes;
vi sortRuntimes;

/*
 For debugging.
 */

#define assertm(exp, msg) assert(((void)msg, exp))

/*
 For profiling.
 */

double accRuntimeRank = 0;
double accRuntimeBreed = 0;
double accRuntimeMutate = 0;

double runtimeSolve = 0;


TravellingSalesmanProblem::TravellingSalesmanProblem(const int problem_size, float* cities,
        const int population_count, const int elite_size, const int mutation_rate, const int verbose) {
    
    this->verbose = verbose;
    this->problem_size = problem_size;
    this->population_count = population_count;
    this->elite_size = elite_size;
    this->mutation_rate = mutation_rate;
    //this->fitness = vector<double>(population_count, 0.0);
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
    //this->population = new Int[population_count * problem_size];
    
    // start SIMD version
    
    /*
     Allocate size bytes of uninitialized storage whose alignment is specified by alignment.
     The size parameter must be an integral multiple of alignment.
     */
    
    int correctedSize = population_count * problem_size * sizeof(Int);
    correctedSize = correctedSize - correctedSize % 32 + 32;
    this->population = (Int*)aligned_alloc(32, correctedSize);
    correctedSize = population_count * problem_size * sizeof(Int);
    correctedSize = correctedSize - correctedSize % 32 + 32;
    this->temp_population = (Int*)aligned_alloc(32, correctedSize);
    //this->mask = (Int*)aligned_alloc(32, problem_size*sizeof(*mask));
    correctedSize = population_count * sizeof(Real);
    correctedSize = correctedSize - correctedSize % 32 + 32;
    this->fitness = (Real*)aligned_alloc(32, correctedSize);
    // end SIMD version
    
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
    
    cout << "size of one gene element is " << sizeof(Int) << " bytes " << endl;
    cout << "size of entire gene is " << sizeof(Int) * problem_size << " bytes " << endl;
    cout << "size of entire population is (w/o fitness) " << sizeof(Int) * problem_size * population_count << " bytes " << endl;
    cout << "size of working set is (w/o fitness) " << sizeof(Int) * problem_size * population_count * 2 << " bytes " << endl;
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
    
    cout << "(total runtime rnd+chunk+split: " << accRnd+accChunk+accSplit << ")" << endl;
    
    double accSumAndMin = (double)accumulate(sumAndMinRuntimes.begin(), sumAndMinRuntimes.end(), 0) / (double)1e9;
    double accSort = (double)accumulate(sortRuntimes.begin(), sortRuntimes.end(), 0) / (double)1e9;
    
    cout << "accumulated runtime rank: " << accRuntimeRank / (double)1e9 << "s" << endl;
    cout << "accumulated runtime breed: " << accRuntimeBreed / (double)1e9 << "s" << endl;
    cout << "accumulated runtime mutate: " << accRuntimeMutate / (double)1e9 << "s" << endl;
    
    cout << "--------------------------" << endl;
    cout << "accumulated runtime sum and min: " << accSumAndMin << "s" << endl;
    cout << "accumulated runtime sort: " << accSort << "s" << endl;
    cout << "--------------------------" << endl;
    
    cout << "(total accumulated runtime: " << (accRuntimeRank+accRuntimeBreed+accRuntimeMutate) / (double)1e9 << ")" << endl;
    
    cout << "runtime SOLVE: " << runtimeSolve / (double)1e9 << endl;
    
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
    
    // start SIMD
    delete this->population;
    delete this->temp_population;
    //delete this->mask;
    delete this->fitness;
    // end SIMD
    
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
    
    hrTime tStart, tEnd;
        
    // Compute fitness
    // TODO: Profiling start
    tStart = myClock.now();
    this->rank_individuals();
    tEnd = myClock.now();
    accRuntimeRank += std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
    // TODO: Profiling end
    
    // Breed children
    // TODO: Profiling start
    tStart = myClock.now();
    this->breed_population();
    tEnd = myClock.now();
    accRuntimeBreed += std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
    // TODO: Profiling end
    
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

    // TODO: PROFILING start
    tStart = myClock.now();
    this->mutate_population();
    tEnd = myClock.now();
    accRuntimeMutate += std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
    // TODO: PROFILING end
    
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

Real TravellingSalesmanProblem::solve(const int nr_epochs, const int rank) {

/*#ifdef debug
    this->rank_individuals();
    for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            // cout << this->population[i+ this->population_count*j] << " ";
        }
        // cout << "\tfit: " << this->fitness[i] << endl;
    }
#endif*/
    
    // TODO: Profiling start
    hrTime tStart, tEnd;
    tStart = myClock.now();
    // TODO: Profiling end
    

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
            //this->logger->log_all_fitness_per_epoch(this->evolutionCounter, this->fitness);
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
    
    // TODO: profiling start
    tEnd = myClock.now();
    runtimeSolve += std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
    // TODO: profiling end

    return this->fitness_best;
}

/*
 Infrastructure for high performance sorting.
 */


#define MAX_REAL std::numeric_limits<typeof(this->fitness_best)>::max()


void TravellingSalesmanProblem::rank_individuals() {
    this->logger->LOG_WC(RANK_INDIVIDUALS_BEGIN);
    
    // TODO: profiling part
    hrTime tStart, tEnd;
    int delta;
    
    tStart = myClock.now();
    // TODO: profiling part
    
    
    /*this->fitness_sum = 0.0;
    this->fitness_best = MAX_REAL;
    
    // min over range can be done efficiently using SIMD
    // sum of range can be done efficiently using SIMD
    // could even unroll loop to break sequential debendencies of "+"
    
    for (int i = 0; i < this->population_count; ++i) {
        double new_fitness = this->evaluate_fitness(i);
        this->fitness[i] = new_fitness;
        this->fitness_sum += new_fitness;
        this->fitness_best = min((double)this->fitness_best, (double)new_fitness);
    }*/
    
    
    // TODO: begin SIMD version
    Real sumFitness = 0.0; // results to compute
    Real minFitness = MAX_REAL;
    
    __m256 fitnessSegSIMD;
    __m256 sumSIMD = _mm256_set1_ps(0.0); // set to zero
    __m256 minSIMD = _mm256_set1_ps(MAX_REAL); // set to MAX_REAL
    
    
    int fitnessIdx = 0;
        
    for (; fitnessIdx <= this->population_count - 8; fitnessIdx = fitnessIdx + 8) {
        
        // 8-fold unrolling
        // make sure execution units are busy
        fitnessSegSIMD = _mm256_load_ps(&(this->fitness[fitnessIdx]));
        
        sumSIMD = _mm256_add_ps(sumSIMD, fitnessSegSIMD);
        minSIMD = _mm256_min_ps(minSIMD, fitnessSegSIMD);
    }
        
    for(; fitnessIdx < this->population_count; fitnessIdx++) { // finish of residual
        sumFitness += fitness[fitnessIdx];
        minFitness = min((double)minFitness,
                         (double)fitness[fitnessIdx]); // because min(.) compares doubles
    }
        
    // compute horizontal sum
    sumSIMD = _mm256_hadd_ps(sumSIMD, sumSIMD); // sums of 2 candidates
    sumSIMD = _mm256_hadd_ps(sumSIMD, sumSIMD); // sums of 4 candidates
    
    Real sumLowH = _mm256_cvtss_f32(sumSIMD);
    sumFitness += sumLowH;
    
    Real sumHighH = _mm256_cvtss_f32(_mm256_permute2f128_ps(sumSIMD, sumSIMD, 1));
    sumFitness += sumHighH;
    
    // compute horizontl min
    __m256 minPermSIMD = _mm256_permute2f128_ps(minSIMD, minSIMD, 0x11);
    minSIMD = _mm256_min_ps(minSIMD, minPermSIMD); // lower half contains 4 candidates
    minPermSIMD = _mm256_shuffle_ps(minSIMD, minSIMD, _MM_SHUFFLE(1, 0, 3, 2)); // _MM_SHUFFLE(1, 0, 3, 2) 0b01001110
    minSIMD = _mm256_min_ps(minSIMD, minPermSIMD); // lower half contains 2 candidates
    minPermSIMD = _mm256_shuffle_ps(minSIMD, minSIMD, _MM_SHUFFLE(3, 2, 0, 1)); // _MM_SHUFFLE(3, 2, 0, 1) 0b11100001
    minSIMD = _mm256_min_ps(minSIMD, minPermSIMD); // lower half contains 1 candidate
    
    Real minFitnessH = _mm256_cvtss_f32(minSIMD);
    minFitness = minFitnessH < minFitness ? minFitnessH : minFitness;
    
    //assertm(abs((sumFitness - this->fitness_sum) / this->fitness_sum) < 1e-5, "fitness sum incorrect");
    //assertm(abs(minFitness - this->fitness_best) < 1e-5, "fitness min incorrect");
        
    this->fitness_best = minFitness;
    this->fitness_sum = sumFitness;
    
    // TODO: end SIMD version
    
    //cout << "after segfault" << endl;
    
    // TODO: profiling part
    tEnd = myClock.now();
    delta = std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
    sumAndMinRuntimes.push_back(delta);
    
    tStart = myClock.now();
    // TODO: profiling part
    
    iota(this->ranks, this->ranks + this->population_count, 0); // fill with ascending values
    
    sort(this->ranks, this->ranks + this->population_count, [this] (int i, int j) { // sort according to fitness
       return this->fitness[i] < this->fitness[j];
    });
    
    // TODO: profiling part
    tEnd = myClock.now();
    delta = std::chrono::duration_cast<hrNanos>(tEnd - tStart).count();
    sortRuntimes.push_back(delta);
    // TODO: profiling part
    
    this->logger->LOG_WC(RANK_INDIVIDUALS_END);
}

Real TravellingSalesmanProblem::evaluate_fitness(const int individual) {
    Real route_distance = 0.0;
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
    
    // start SIMD
    // TODO: make sure child is aligned to 32
    // end SIMD
    
    bool useSIMD = false;
    
    
    if (useSIMD) {
        
        // TODO: align population
        
        // TODO: sample multiple random integers 2x
        // TODO: use min, max
        // leave this out for the moment as it seems to be much effort
        int geneA = this->rand_range(0, this->problem_size - 1);
        int geneB = this->rand_range(0, this->problem_size - 1);
        int startGene = min(geneA, geneB);
        int endGene = max(geneA, geneB);
        
        // peel
        int idx = startGene;
        while(idx % 8 != 0 && idx <= endGene) {
            child[idx] = this->population[parent1*this->problem_size + idx];
            
            idx = idx + 1;
        }
        // chunks
        __m256i simdChunk;
        while(idx+8 <= endGene) {
            simdChunk = _mm256_load_si256((__m256i *)&(this->population[parent1]));
            _mm256_store_si256 ((__m256i *)&child[idx], simdChunk);
            
            idx = idx + 8;
        }
        // peel
        while(idx <= endGene) {
            child[idx] = this->population[parent1*this->problem_size + idx];
            
            idx = idx + 1;
        }
        
        
                
        // TODO: extract subarray
        // masking maybe?
        
        // TODO: extract disjoint indices
        
        
    } else {
        
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
            
            int mask[problem_size]; // size of a single individual
            for(int idx = 0; idx < problem_size; idx++) {
                mask[idx] = 0xFFFF;
            }
            // cities are indexed 0, ..., (problem_size-1)
            
            // start SIMD version
            
            // avx2 needs array addresses aligned to 32
            
            /*int ones = 0xFFFF; // broadcast 0xFFFF to eight 32-bit integers
            __m256i onesSIMD = _mm256_set1_epi32(ones);
            
            int maskIdx;
            for(maskIdx = 0; maskIdx <= problem_size - 8; maskIdx = maskIdx + 8) {
                _mm256_store_si256((__m256i *)&mask[maskIdx], onesSIMD);
            }
            
            for(; maskIdx < problem_size; maskIdx++) {
                mask[maskIdx] = ones;
            }*/
            // end SIMD version
            
            
            for (int i = startGene; i <= endGene; ++i) {
                child[i] = POP(parent1, i);
                //assertm(0 <= POP(parent1, i) && POP(parent1, i) <= problem_size-1, "segfault mask, loop chunk");
                //assertm(mask[POP(parent1, i)] == 1, "gene part is already masked, loop chunk");
                mask[POP(parent1, i)] = 0x0000; // no comparisons
            }
            
            // start SIMD version
            /*int zeros = 0x0000;
            __m256i zerosSIMD = _mm256_set1_epi32(zeros);
            
            int geneIdx;
            __m256i geneSegSIMD;
            for(geneIdx = startGene; geneIdx <= (endGene - 7); geneIdx = geneIdx + 8) {
                // by (endGene - 7) we also take endGene
                
                // TODO: genes are in general not "int" data type
                // this will overshoot and maybe do other bad things
                geneSegSIMD = _mm256_load_si256((__m256i *)
                                    &(this->population[parent1*problem_size + geneIdx]));
                _mm256_store_si256((__m256i *)child[geneIdx], geneSegSIMD);
                
                _mm256_store_si256(&mask[geneIdx], zerosSIMD);
            }
            
            Int tmp;
            for(; geneIdx <= endGene; geneIdx++) {
                tmp = this->population[parent1*problem_size + geneIdx];
                child[geneIdx] = tmp;
                mask[tmp] = zeros;
            }*/
            // end SIMD version
            
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
        
            // writing all the time to child works better
            // this could be mapped to SIMD in a straightforward way
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
            
            
            // start SIMD version
            /*int parent2idx = 0;
            
            int childIdx = 0;
            __m256i parentSegSIMD;
            __m256i maskSegSIMD;
            __m256i maskAddrSIMD = _mm256_set1_epi32(mask);
            __m256i childSegSIMD;
            for(; childIdx <= startGene - 8; childIdx = childIdx + 8) {
                // TODO: this should work well
                // idea is that consecutive child indices get consecutive parent indices
                // parent indices can be collectively bit shift operation transformed somehow
                // need an unaligned load
                // maybe need to permute
                // maybe it's possible to do a bit less overwriting (-> less data transfer)
                
                // BEGIN LOOP
                
                parentSegSIMD = _mm256_load_si256(
                                    (__m256i *)&(this->population[parent2*problem_size + parent2idx]));
                
                // load mask with unaligned load or bcast
                maskSegSIMD = _mm256_i32gather_epi32(mask, parentSegSIMD, 1);
                // dst = base_addr + offset
                
                parentSegSIMD = _mm256_and_si256(parentSegSIMD, maskSegSIMD);
                
                // shuffle
                // blend - for mixing
                
                // could calculate parent increment based on this
                
                
                childSegSIMD = _mm256_mask_i32gather_epi32(childSegSIMD, mask, parentSegSIMD, maskSegSIMD, 1);
                // dst = (src, base_addr, offset, mask, 1)
                // dst = src & (not mask) + (base_addr + offset) & mask
                
                // load
                // unaligned load
                
                // END LOOP
                
                // write child chunk
                
            }
            
            for(; childIdx < startGene; childIdx++) {
                child[childIdx] = POP(parent2, parent2idx);
                childIdx = childIdx + (1 & mask[POP(parent2, parent2idx)]);
                parent2idx++;
            }*/
            // end SIMD version
            
            
            /*
             introduce city -> index map
             in addition to index -> city map
             */
            
            
            /*int parent2idx = 0;
            
            for(int childIdx = 0; childIdx < startGene; childIdx++) {
                while(mask[POP(parent2, parent2idx)] == 1) {
                    parent2idx++;
                }
                child[childIdx] = POP(parent2, parent2idx);
                parent2idx++;
            }
            
            for(int childIdx = endGene+1; childIdx < problem_size; childIdx++) {
                while(mask[POP(parent2, parent2idx)] == 1) {
                    parent2idx++;
                }
                child[childIdx] = POP(parent2, parent2idx);
                parent2idx++;
            }*/
            
            
            /*int childIdx = 0;
            
            for(int parent2idx = 0; parent2idx < problem_size; parent2idx++) { // iterate over parent2
                
                if(childIdx == startGene) {
                    childIdx = endGene + 1;
                    if(childIdx == problem_size) break;
                    // because this can fail if startGene == endGene == problem_size-1
                }
                
                //assertm(0 <= POP(parent2, parent2idx) && POP(parent2, parent2idx) <= problem_size-1, "segfault mask, loop split");
                if(mask[POP(parent2, parent2idx)] == 1) {
                    //cout << childIdx << endl;
                    //assertm(0 <= childIdx && childIdx <= problem_size-1, "segfault child");
                    child[childIdx] = POP(parent2, parent2idx);
                    childIdx++;
                }
                
            }*/
            
            
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
        
    } // end if else
    
}

void TravellingSalesmanProblem::breed_population() {
    this->logger->LOG_WC(BREED_POPULATION_BEGIN);
    
    // TODO: do this in-place
    // TODO: make this global (don't allocate, free, allocate, free, ... as this is "huge")
    // in order to speed things up
    //Int temp_population[this->population_count][this->problem_size];
    

    // Keep the best individuals
    for (int i = 0; i < this->elite_size; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            //temp_population[i][j] = POP(this->ranks[i], j);
            // start SIMD version
            temp_population[i*problem_size + j] = POP(this->ranks[i], j);
            // end SIMD version
            //population[i][j] = POP(this->ranks[i], j);
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
        //this->breed(rand1, rand2, temp_population[i]);
        // start SIMD version
        this->breed(rand1, rand2, &temp_population[i*problem_size]);
        // end SIMD version
        //this->breed(rand1, rand2, &(this->population[i*problem_size]));
    }
    
    /*for (int i = 0; i < this->population_count; ++i) {
        for (int j = 0; j < this->problem_size; ++j) {
            VAL_POP(i, j);
            POP(i, j) = temp_population[i][j];
        }
    }*/
    this->logger->LOG_WC(BREED_POPULATION_END);
    
}

void TravellingSalesmanProblem::mutate(const int individual) {
    
    bool useSIMD = false;
    
    
    if(useSIMD) {
        
        __m256i simdMutationRate = _mm256_set1_epi32(this->mutation_rate);
        
        // TODO: random simd vector
        // faster check  i
        
        // unaligned load & unaligned store
        
        
    } else {
    
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

Real TravellingSalesmanProblem::getFitness(int indivIdx) { // for Island
    return fitness[indivIdx];
}

void TravellingSalesmanProblem::setFitness(int indivIdx, Real newFitness) { // for Island
    fitness[indivIdx] = newFitness;
}

Real TravellingSalesmanProblem::getMinFitness() { // for Island
    //return *min_element(fitness.begin(), fitness.end());
    return fitness_best;
}
