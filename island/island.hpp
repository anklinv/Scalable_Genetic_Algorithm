#ifndef DPHPC_PROJECT_ISLAND_HPP
#define DPHPC_PROJECT_ISLAND_HPP

#include "mpi.h" /* requirement for MPI */

#include <algorithm>
#include <iostream> // for debugging
#include <limits>

#include "../sequential/travelling_salesman_problem.hpp"


using namespace std;

// set up send buffer
// set up receive buffer
// ---- SYNCHRONOUS exchange of data (needs send buffer & receive buffer) ----
// integrate data

// ASYNCHRONOUS
// ?

class Island {
    
public:
    
    /**
     An Island wraps around a TravellingSalesmanProblem
     - Fitness evaluation (C_eval) , cross-over and mutation (C_oper) are done by the underlying TSP
     - The Island adds communication (C_comm)  and performs selection and replacement in this context (C_oper)
     
     Because of the migration across islands it is necessary that all ranks in MPI_COMM_WORLD call the constructor
     with the same settings (i.e. run the same code).
     
     \param tsp a pointer to a fully initialized TSP
     \param migrationPeriod the amount of iterations between two migration steps
     \param migrationAmount the number of individuals each island sends to all others
     \param numPeriods numPeriods * migrationPeriod yields the total number of iterations
    */
    Island(TravellingSalesmanProblem* tsp,
           int migrationPeriod, int migrationAmount, int numPeriods):
    tsp(tsp),
    migrationPeriod(migrationPeriod),
    migrationAmount(migrationAmount),
    numPeriods(numPeriods) {}
    
    /**
     Executes the GA on the current rank. Because of MPI_Allgather it is necessary that all ranks in MPI_COMM_WORLD
     execute the GA simultaneously.
     \return the length of the shortest path found by the algorithm
     */
    double solve();
    
    
    enum MigrationTopology {
        FULLY_CONNECTED,
        ISOLATED,
        RING
    };

    enum SelectionPolicy {
        TRUNCATION, // ok
        FITNESS_PROPORTIONATE_SELECTION, // ok
        STOCHASTIC_UNIVERSAL_SAMPLING, // ok
        TOURNAMENT_SELECTION, // ok
        PURE_RANDOM // ok
    };

    enum ReplacementPolicy {
        TRUNCATION, // ok
        DEJONG_CROWDING, // ok
        PURE_RANDOM // ok
    };
    
    
private:
    
    /// Helps sorting indices of individuals according to fitness
    typedef struct Individual {
        
        int idx;
        double fitness;
        
        bool operator<(const Individual& other) const {
            return (this->fitness) < other.fitness;
        }
        
    } Individual;
    
    
    /// Use a pointer to avoid dealing with object initialization
    const TravellingSalesmanProblem* TSP;
 
    
    /// Migration topology (connections between islands)
    const MigrationTopology MIGRATION_TOPOLOGY;

    /// Selection policy (source island)
    const SelectionPolicy SELECTION_POLICY;
    
    /// Replacement policy (destination island)
    const ReplacementPolicy REPLACEMENT_POLICY;
    
    
    /**
     The amount of iterations of the algorithm between two migration steps
     */
    const int MIGRATION_PERIOD;
    
    /**
     The amount of individuals each island sends to all other islands
     */
    const int MIGRATION_AMOUNT;
    
    /**
     The number of migration periods. The overall number of iterations is given by numPeriods * migrationPeriod.
     */
    const int NUM_PERIODS;
    
    
    /// Replaces the geneSize entries of oldGene with the geneSize entries of newGene. This corresponds to replacing
    /// an individual (essentially a gene and a fitness value based thereon) of a population with a new one.
    void overwriteGene(int* newGene, int* oldGene, int geneSize);
    
    /// Computes the Hamming distance between two genes. Each path (gene) is mapped to a list of cities starting
    /// with the city indexed 1. The Hamming distance is computed as the amount of list indices where the list elements
    /// are different.
    /// approach to compute the Hamming Distance
    /// search for city 1 and then to a cyclic comparison - 1 is safer than 0 because it should work for indexing starting at 0
    /// as well as for indexing starting at 1
    /// as the approach is cyclic the geneSize is needed for wrapping around
    int computeHammingDistance(int* firstGene, int* scndGene, int geneSize);
    
    
    /// Does fitness proportionate selection (roulette wheel selection). The selected individuals are drawn from a probability
    /// distribution where the individuals are weighted with their fitness values. The fitness values do not have to be sorted. This
    /// function does several O(n) linear passes over the fitness values where n is the population size. Returns the indices of
    /// the numIndividualsToSample selected individuals as array.
    int* fitnessProportionateSelection(double* fitness, int numIndividuals, // TODO: change access pattern to these variables
                                       int numIndividualsToSample);
        
    /// Does stochastic universal sampling. This function does two O(n) linear passes over the fitness values where n is the
    /// population size. Returns the indices of the numIndividualsToSample selected individuals as array.
    int* stochasticUniversalSampling(double* fitness, int numIndividuals, // TODO: change access pattern to these variables
                                     int numIndividualsToSample, int* sampledIndividuals);
    
    /// Does tournament selection. Returns the indices of numIndividualsToSample selected individuals as array. For each
    /// individual to be selected, tournamentSize individuals are sampled uniformly at random and the best individual thereof
    /// is selected. Standard values for tournamentSize are 2 or 3.
    int* tournamentSelection(double* fitness, int numIndividuals, // TODO: change access pattern to these variables
                             int tournamentSize, int numIndividualsToSample);
    
    /// Returns the indices of the numIndividualsToSample best individuals as an array. The fitness values are assumed to be sorted in
    /// ascending order. Lower fitness is assumed to be better.
    int* truncationSelection(int numIndividualsToSample); // TODO: add access to the ranks
    
    /// Returns the indices of numIndividualsToSample randomly chosen individuals. The individuals are sampled uniformly at random.
    /// A specific individual can be selected multiple times.
    int* pureRandomSelection(int numIndividuals, // TODO: change access to this
                             int numIndividualsToSample)
    
    /// geneSize is numNodes of the TSP graph
    /// islandSize is the number of individuals at the island
    /// ranks: individuals are assumed to be sorted. 0 is best, islandSize-1 is worst
    void truncationReplacement(const TravellingSalesmanProblem* tsp, int geneSize, int islandSize, // TODO: change access to this. Use the tsp pointer.
                               int numImmigrants, int** immigrantGenes, double* immigrantFitnesses)
    
    /// All individuals to be replaced are chosen uniformly at random. It is possible that an immigrant is itself replaced
    /// by a subsequent one.
    void pureRandomReplacement(int islandSize, int geneSize, // TODO: change access to these variables
                               int numImmigrants, int** immigrantGenes, double* immigrantFitnesses)
    
    /// Uses crowding for replacement. For each immigrant, crowdSize individuals are sampled uniformly at random
    /// and their Hamming Distance to the immigrant is computed. The individual which has the smallest Hamming
    /// Distance to the immigrant is replaced. It is possible that an immigrant is itself replaced by a subsequent one.
    void crowdingReplacement(int geneSize, int islandSize, // TODO: change access to these variables
                             int crowdSize,
                             int numImmigrants, double* immigrantFitnesses, int** immigrantGenes)
    
};

#endif //DPHPC_PROJECT_ISLAND_HPP
