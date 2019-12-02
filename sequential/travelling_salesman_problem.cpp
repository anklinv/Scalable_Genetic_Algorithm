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
#include <bitset> // for debugging
#include <iomanip>
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


TravellingSalesmanProblem::TravellingSalesmanProblem(const int problem_size, Real* cities,
        const int population_count, const int elite_size, const int mutation_rate, const int verbose) {
    
    this->verbose = verbose;
    this->problem_size = problem_size;
    this->population_count = population_count;
    this->elite_size = elite_size;
    this->mutation_rate = mutation_rate;
    //this->fitness = vector<double>(population_count, 0.0);
    this->ranks = new int[population_count];
    // TODO: this is also quite a chunk of data
    this->cities = cities;
    // TODO: this is also quite a chunk of data
    random_device rd;
    this->gen = mt19937(rd());

    this->log_iter_freq = 100;

    // Initialize fields to be initialized later
    this->logger = nullptr;
    this->fitness_best = -1;
    this->fitness_sum = -1;

    // TODO: make this nicer
    //this->population = new Int[population_count * problem_size];
    
    
    
    //cout << std::hex; // TODO: remove this
    
    // start SIMD version
    
    /*
     Allocate size bytes of uninitialized storage whose alignment is specified by alignment.
     The size parameter must be an integral multiple of alignment.
     */
    
    // TODO: align cities data
    
    int correctedSize = population_count * problem_size * sizeof(Int);
    correctedSize = correctedSize - correctedSize % 32 + 32;
    this->population = (Int*)aligned_alloc(32, correctedSize);
    
    correctedSize = population_count * problem_size * sizeof(Int);
    correctedSize = correctedSize - correctedSize % 32 + 32;
    this->temp_population = (Int*)aligned_alloc(32, correctedSize);
    
    correctedSize = problem_size * sizeof(Int); // mask is Int
    correctedSize = correctedSize - correctedSize % 32 + 32;
    this->mask = (Int*)aligned_alloc(32, correctedSize);
    
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
    
    double accSumAndMin = (double)accumulate(sumAndMinRuntimes.begin(), sumAndMinRuntimes.end(), 0) / (double)1e9;
    double accSort = (double)accumulate(sortRuntimes.begin(), sortRuntimes.end(), 0) / (double)1e9;
    
    cout << "accumulated runtime rank: " << accRuntimeRank / (double)1e9 << "s" << endl;
    cout << "accumulated runtime breed: " << accRuntimeBreed / (double)1e9 << "s" << endl;
    cout << "(total runtime rnd+chunk+split: " << accRnd+accChunk+accSplit << ")" << endl;
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
    delete this->mask;
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
    
    // TODO: begin scalar version
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
    // TODO: end scalar version
    
    // TODO: begin SIMD version
    for(int indivIdx = 0; indivIdx < population_count; indivIdx++) {
        fitness[indivIdx] = evaluate_fitness(indivIdx);
    }
    
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
    
    // compute horizontal min
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
    
    // maybe use "in-place" sort
    // hpc library for sorting or similar
    
    // maybe the simd library thing?
    
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


void print256_bitset(__m256i var) {
    
    uint32_t *val = (uint32_t*) &var;
    
    cout << bitset<32>(val[0]) << " " << bitset<32>(val[1]) << " "
        << bitset<32>(val[2]) << " " << bitset<32>(val[3]) << " "
        << bitset<32>(val[4]) << " " << bitset<32>(val[5]) << " "
        << bitset<32>(val[6]) << " " << bitset<32>(val[7]) << endl;
}


Real TravellingSalesmanProblem::evaluate_fitness(const int individual) {
    
    // size of this part of the working set is:
    // sizeof(Real) * problem_size * problem_size
    
    
    // TODO: begin SIMD version
    // compute how many mask elements are covered by a __m256i
    const int INC_GENE = (256 / 8) / sizeof(Int); // bytes
        
    Real sumDistances = 0;
    __m256 sumDistancesSIMD = _mm256_set1_ps(0);
    
    // indices for distance matrix lookup
    //__m256i geneSegSIMD;
    //__m256i geneSegShiftedSIMD;
    
    const __m256i PROBLEM_SIZE_SIMD = _mm256_set1_epi32(problem_size);
    
    __m256i offsetsSIMD;
    
    // distances of the current gene segment
    // (8 indices means 7 distances)
    // ok ok ok ok ok ok ok garbage
    __m256 distancesSIMD;
    
    __m256i geneSegIdx3MaskSIMD = _mm256_set_epi32(0, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0);
    
    int geneIdx = 0;
    
    for(; geneIdx <= (problem_size - INC_GENE); geneIdx = geneIdx + (INC_GENE - 1)) {
        
        __m256i geneSegSIMD = _mm256_load_si256((__m256i *)&POP(individual, geneIdx));
        
        // shift 128-bit lanes to the left by 32 bit
        // 7 6 5 4 3 2 1 0 => 6 5 4 x 2 1 0 x (x padded with zeros)
        // use extract to recover 3
        int geneSegIdx3 = _mm256_extract_epi32(geneSegSIMD, 3);
        
        __m256i geneSegShiftedSIMD = _mm256_slli_si256(geneSegSIMD, 4);
        
        __m256i geneSegIdx3SIMD = _mm256_set1_epi32(geneSegIdx3);
        geneSegIdx3SIMD = _mm256_and_si256(geneSegIdx3SIMD, geneSegIdx3MaskSIMD);
        
        geneSegShiftedSIMD = _mm256_or_si256(geneSegShiftedSIMD, geneSegIdx3SIMD);
        
        // now we have like like
        // 6 5 4 3 2 1 0 x geneSegShiftedSIMD (x padded with zeros)
        // 7 6 5 4 3 2 1 0 geneSegSIMD
        
        offsetsSIMD = _mm256_mullo_epi32(geneSegShiftedSIMD, PROBLEM_SIZE_SIMD); // super slow
        offsetsSIMD = _mm256_add_epi32(offsetsSIMD, geneSegSIMD);
        
        // make sure cities is aligned to 32 in memory
        distancesSIMD = _mm256_i32gather_epi32(cities, offsetsSIMD, 4);
        
        // 6-7 5-6 4-5 3-4 2-3 1-2 0-1 x
        
        // simultaneous add to SIMD sum
        sumDistancesSIMD = _mm256_add_ps(sumDistancesSIMD, distancesSIMD);
    }
    
    // ok ok ok ok ok ok ok garbage
    // (use mask to eliminate garbage)
    __m256i garbageMaskSIMD = _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                             0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0);
    sumDistancesSIMD = _mm256_and_si256(sumDistancesSIMD, garbageMaskSIMD);
        
    // compute horizontal sum
    sumDistancesSIMD = _mm256_hadd_ps(sumDistancesSIMD, sumDistancesSIMD); // sums of 2 candidates
    sumDistancesSIMD = _mm256_hadd_ps(sumDistancesSIMD, sumDistancesSIMD); // sums of 4 candidates
    
    Real sumLowH = _mm256_cvtss_f32(sumDistancesSIMD);
    sumDistances += sumLowH;
    
    Real sumHighH = _mm256_cvtss_f32(_mm256_permute2f128_ps(sumDistancesSIMD, sumDistancesSIMD, 1));
    sumDistances += sumHighH;
    
    // scalar loop for residual
    for(; geneIdx < (problem_size - 1); geneIdx++) {
        sumDistances += DIST(POP(individual, geneIdx), POP(individual, geneIdx + 1));
    }
    sumDistances += DIST(POP(individual, problem_size-1), POP(individual, 0)); // round trip
    return sumDistances;
    // TODO: end SIMD version
    
    
    // TODO: start sequential version
    /*Real route_distance = 0.0;
    
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
    
    //assertm(sumDistances == route_distance, "computed distance is not correct");
    
    return route_distance;*/
    // TODO: end sequential version
}




//this function takes up the most time in an epoch
//possible solutions:
//  * take 50% of each parent as opposed to randomly taking a sequence
//  *
void TravellingSalesmanProblem::breed(const int parent1, const int parent2, Int* child) {
    
    
    bool useSIMD = false;
    if (useSIMD) {
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
            
            // TODO: start sequential version
            /*int mask[problem_size]; // size of a single individual
            for(int idx = 0; idx < problem_size; idx++) {
                mask[idx] = 1;
            }*/
            // cities are indexed 0, ..., (problem_size-1)
            // TODO: end sequential version
            
            // TODO: start SIMD version
            // compute how many mask elements are covered by a __m256i
            const int INC_MASK = (256 / 8) / sizeof(Int); // bytes
            
            const __m256i ALL_BITS_SET_SIMD = _mm256_set1_epi32(0xFFFFFFFF);
            
            int maskIdx;
            for(maskIdx = 0; maskIdx <= problem_size - INC_MASK; maskIdx = maskIdx + INC_MASK) {
                _mm256_store_si256((__m256i *)(&mask[maskIdx]), ALL_BITS_SET_SIMD);
            }
            
            Int ALL_BITS_SET;
            if(sizeof(Int) == 2) ALL_BITS_SET = 0xFFFF; // 0xFFFF for 16-bit integers
            else if(sizeof(Int) == 4) ALL_BITS_SET = 0xFFFFFFFF; // 0xFFFFFFFF for 32-bit integers
            
            for(; maskIdx < problem_size; maskIdx++) {
                mask[maskIdx] = ALL_BITS_SET;
            }
            // TODO: end SIMD version
            
            // TODO: start sequential version
            /*for (int i = startGene; i <= endGene; ++i) {
                // when running this version it is super important to use the
                // local mask array
                child[i] = POP(parent1, i);
                //assertm(0 <= POP(parent1, i) && POP(parent1, i) <= problem_size-1, "segfault mask, loop chunk");
                //assertm(mask[POP(parent1, i)] == 0xFFFFFFFF, "gene part is already masked, loop chunk");
                mask[POP(parent1, i)] = 0x00000000;
            }*/
            // TODO: end sequential version
            
            // TODO: start SIMD version
            if(sizeof(Int) == 4) {
                
                const int ALL_BITS_ZERO = 0x00000000;
                
                __m256i geneSegSIMD;
                
                int geneIdx;
                
                for(geneIdx = startGene; geneIdx <= (endGene - 7); geneIdx = geneIdx + 8) {
                    // with (endGene - 7) we also take endGene
                    
                    geneSegSIMD = _mm256_load_si256((__m256i *)&POP(parent1, geneIdx));
                    
                    _mm256_store_si256((__m256i *)&child[geneIdx], geneSegSIMD);
                    
                    // extract - store
                    int geneSegIdx0 = _mm256_extract_epi32(geneSegSIMD, 0);
                    mask[geneSegIdx0] = ALL_BITS_ZERO;
                    int geneSegIdx1 = _mm256_extract_epi32(geneSegSIMD, 1);
                    mask[geneSegIdx1] = ALL_BITS_ZERO;
                    int geneSegIdx2 = _mm256_extract_epi32(geneSegSIMD, 2);
                    mask[geneSegIdx2] = ALL_BITS_ZERO;
                    int geneSegIdx3 = _mm256_extract_epi32(geneSegSIMD, 3);
                    mask[geneSegIdx3] = ALL_BITS_ZERO;
                    int geneSegIdx4 = _mm256_extract_epi32(geneSegSIMD, 4);
                    mask[geneSegIdx4] = ALL_BITS_ZERO;
                    int geneSegIdx5 = _mm256_extract_epi32(geneSegSIMD, 5);
                    mask[geneSegIdx5] = ALL_BITS_ZERO;
                    int geneSegIdx6 = _mm256_extract_epi32(geneSegSIMD, 6);
                    mask[geneSegIdx6] = ALL_BITS_ZERO;
                    int geneSegIdx7 = _mm256_extract_epi32(geneSegSIMD, 7);
                    mask[geneSegIdx7] = ALL_BITS_ZERO;
                }
                
                int geneSeg;
                for(; geneIdx <= endGene; geneIdx++) {
                    geneSeg = POP(parent1, geneIdx);
                    child[geneIdx] = geneSeg;
                    mask[geneSeg] = ALL_BITS_ZERO;
                }
            } else if(sizeof(Int) == 2) {
                
                const int ALL_BITS_ZERO = 0x0000;
                
                __m256i geneSegSIMD;
                
                int geneIdx;
                const int INC_GENE = (256 / 8) / sizeof(Int); // bytes
                
                for(geneIdx = startGene; geneIdx <= (endGene - INC_GENE + 1); geneIdx = geneIdx + INC_GENE) {
                    // with (endGene - 7) we also take endGene
                    
                    geneSegSIMD = _mm256_load_si256((__m256i *)&POP(parent1, geneIdx));
                    
                    _mm256_store_si256((__m256i *)&child[geneIdx], geneSegSIMD);
                    
                    // extract - store
                    Int geneSegIdx0 = _mm256_extract_epi16(geneSegSIMD, 0);
                    mask[geneSegIdx0] = ALL_BITS_ZERO;
                    Int geneSegIdx1 = _mm256_extract_epi16(geneSegSIMD, 1);
                    mask[geneSegIdx1] = ALL_BITS_ZERO;
                    Int geneSegIdx2 = _mm256_extract_epi16(geneSegSIMD, 2);
                    mask[geneSegIdx2] = ALL_BITS_ZERO;
                    Int geneSegIdx3 = _mm256_extract_epi16(geneSegSIMD, 3);
                    mask[geneSegIdx3] = ALL_BITS_ZERO;
                    Int geneSegIdx4 = _mm256_extract_epi16(geneSegSIMD, 4);
                    mask[geneSegIdx4] = ALL_BITS_ZERO;
                    Int geneSegIdx5 = _mm256_extract_epi16(geneSegSIMD, 5);
                    mask[geneSegIdx5] = ALL_BITS_ZERO;
                    Int geneSegIdx6 = _mm256_extract_epi16(geneSegSIMD, 6);
                    mask[geneSegIdx6] = ALL_BITS_ZERO;
                    Int geneSegIdx7 = _mm256_extract_epi16(geneSegSIMD, 7);
                    mask[geneSegIdx7] = ALL_BITS_ZERO;
                    
                    Int geneSegIdx8 = _mm256_extract_epi16(geneSegSIMD, 8);
                    mask[geneSegIdx8] = ALL_BITS_ZERO;
                    Int geneSegIdx9 = _mm256_extract_epi16(geneSegSIMD, 9);
                    mask[geneSegIdx9] = ALL_BITS_ZERO;
                    Int geneSegIdx10 = _mm256_extract_epi16(geneSegSIMD, 10);
                    mask[geneSegIdx10] = ALL_BITS_ZERO;
                    Int geneSegIdx11 = _mm256_extract_epi16(geneSegSIMD, 11);
                    mask[geneSegIdx11] = ALL_BITS_ZERO;
                    Int geneSegIdx12 = _mm256_extract_epi16(geneSegSIMD, 12);
                    mask[geneSegIdx12] = ALL_BITS_ZERO;
                    Int geneSegIdx13 = _mm256_extract_epi16(geneSegSIMD, 13);
                    mask[geneSegIdx13] = ALL_BITS_ZERO;
                    Int geneSegIdx14 = _mm256_extract_epi16(geneSegSIMD, 14);
                    mask[geneSegIdx14] = ALL_BITS_ZERO;
                    Int geneSegIdx15 = _mm256_extract_epi16(geneSegSIMD, 15);
                    mask[geneSegIdx15] = ALL_BITS_ZERO;
                }
                
                int geneSeg;
                for(; geneIdx <= endGene; geneIdx++) {
                    geneSeg = POP(parent1, geneIdx);
                    child[geneIdx] = geneSeg;
                    mask[geneSeg] = ALL_BITS_ZERO;
                }
            }
            // TODO: end SIMD version
            
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
            
            // TODO: start SIMD version
            /*__m256i maskSegSIMD;
            int childIdx = 0;
            int parent2idx = 0;
            while(true){
                
                geneSegSIMD = _mm256_load_si256((__m256i *)&POP(parent2, parent2idx));
                parent2idx = parent2idx + 8;
                if(parent2idx + 8 > problem_size) break;
                
                maskSegSIMD = _mm256_i32gather_epi32(mask, geneSegSIMD, 4);
                
                unsigned int ctl = _mm256_movemask_epi8(maskSegSIMD) & 0x88888888;
                
                uint64_t expanded_mask = _pdep_u64(ctl, 0x0101010101010101);
                expanded_mask *= 0xFF;
                const uint64_t identity_indices = 0x0706050403020100;
                uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

                __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
                __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);

                _mm256_permutevar8x32_ps(geneSegSIMD, shufmask);
                
                
                int numSetBits = __builtin_popcount(ctl);
                
                _mm256_store_si256((__m256i *)(&child[childIdx]), geneSegSIMD);
                
                childIdx = childIdx + numSetBits;
                
            }*/
            // TODO: end SIMD version
            
            
            // TODO: start SIMD version
            /*// reuse geneSegSIMD
            __m256i maskSegSIMD;
            
            int childIdx = 0; // this is used for write synch
            if(startGene == 0) {
                // ensure that childIdx is valid
                childIdx = endGene + 1;
            }
            
            int parent2idx = 0;
            
            for(; parent2idx <= problem_size - 8; parent2idx = parent2idx + 8) {
                
                // load gene segment
                geneSegSIMD = _mm256_load_si256((__m256i *)&POP(parent2, parent2idx));
                // load corresponding mask segment
                maskSegSIMD = _mm256_i32gather_epi32(mask, geneSegSIMD, 4);
                
                
                int ctl = _mm256_movemask_epi8(maskSegSIMD);
                
                if(ctl & 0x00000008) {
                    child[childIdx] = _mm256_extract_epi32(geneSegSIMD, 0);
                    childIdx = (childIdx+1)<startGene||endGene<(childIdx+1)?(childIdx+1):(endGene+1); // write synch
                }
                if(ctl & 0x00000080) {
                    child[childIdx] = _mm256_extract_epi32(geneSegSIMD, 1);
                    childIdx = (childIdx+1)<startGene||endGene<(childIdx+1)?(childIdx+1):(endGene+1);
                }
                if(ctl & 0x00000800) {
                    child[childIdx] = _mm256_extract_epi32(geneSegSIMD, 2);
                    childIdx = (childIdx+1)<startGene||endGene<(childIdx+1)?(childIdx+1):(endGene+1);
                }
                if(ctl & 0x00008000) {
                    child[childIdx] = _mm256_extract_epi32(geneSegSIMD, 3);
                    childIdx = (childIdx+1)<startGene||endGene<(childIdx+1)?(childIdx+1):(endGene+1);
                }
                if(ctl & 0x00080000) {
                    child[childIdx] = _mm256_extract_epi32(geneSegSIMD, 4);
                    childIdx = (childIdx+1)<startGene||endGene<(childIdx+1)?(childIdx+1):(endGene+1);
                }
                if(ctl & 0x00800000) {
                    child[childIdx] = _mm256_extract_epi32(geneSegSIMD, 5);
                    childIdx = (childIdx+1)<startGene||endGene<(childIdx+1)?(childIdx+1):(endGene+1);
                }
                if(ctl & 0x08000000) {
                    child[childIdx] = _mm256_extract_epi32(geneSegSIMD, 6);
                    childIdx = (childIdx+1)<startGene||endGene<(childIdx+1)?(childIdx+1):(endGene+1);
                }
                if(ctl & 0x80000000) {
                    child[childIdx] = _mm256_extract_epi32(geneSegSIMD, 7);
                    childIdx = (childIdx+1)<startGene||endGene<(childIdx+1)?(childIdx+1):(endGene+1);
                }
                
            }
            
            // reuse geneSeg
            for(; parent2idx < problem_size; parent2idx++) {
                geneSeg = POP(parent2, parent2idx);
                if(mask[geneSeg]) {
                    child[childIdx] = geneSeg;
                    childIdx = (childIdx+1)<startGene||endGene<(childIdx+1)?(childIdx+1):(endGene+1);
                }
            }*/
            // TODO: end SIMD version
            
            
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
                cout << "problem size is " << problem_size << endl;
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
    } else {
        
        if (rand() % this->mutation_rate == 0) {
            int swap = rand_range(0, this->problem_size - 1);
            int swap_with = rand_range(0, this->problem_size - 1);

            VAL_POP(individual, swap);
            VAL_POP(individual, swap_with);
            Int city1 = POP(individual, swap);
            Int city2 = POP(individual, swap_with);
            POP(individual, swap) = city2;
            POP(individual, swap_with) = city1;
        }
    
        /*if (rand() % this->mutation_rate == 0) {
            int swap = rand_range(0, this->problem_size - 1);
            int swap_with = rand_range(0, this->problem_size - 1);

            VAL_POP(individual, swap);
            VAL_POP(individual, swap_with);
            int city1 = POP(individual, swap);
            int city2 = POP(individual, swap_with);
            POP(individual, swap) = city2;
            POP(individual, swap_with) = city1;
        }*/
    
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
