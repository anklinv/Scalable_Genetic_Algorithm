#ifndef DPHPC_PROJECT_ISLAND_HPP
#define DPHPC_PROJECT_ISLAND_HPP

#include "mpi.h" /* requirement for MPI */

#include <algorithm>
#include <iostream> // for debugging
#include <limits>

#include "../sequential/travelling_salesman_problem.hpp"


using namespace std;


// TODO: ASYNCHRONOUS
// e.g. loop with periodic checking
// e.g. nonblocking send and receive with periodic checking
// cf lecture


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
        FULLY_CONNECTED, // ok
        ISOLATED, // ok
        RING // 1D directed grid - ok
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
        PURE_RANDOM, // ok
        DEJONG_CROWDING // ok
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
    // TODO: change to object variable
    const TravellingSalesmanProblem* TSP;
 
    
    /// Migration topology (connections between islands)
    const MigrationTopology MIGRATION_TOPOLOGY;
    
    /// Number of immigrants each island receives during each migration step. This is necessary to
    /// fully specify the migration topology.
    const int NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION;
    
    /// Number of active islands. Has to be equal to the number of ranks in MPI_COMM_WORLD.  It is
    /// necessary that all ranks in MPI_COMM_WORLD execute the GA simultaneously.
    const int NUM_ACTIVE_ISLANDS; // TODO: replace this with SIZE_COMM_WORLD or similar
    
    /// The ID of the rank (process) in which the Island is created
    const int RANK_ID;
    

    /// Selection policy (source island)
    const SelectionPolicy SELECTION_POLICY;
    
    /// Replacement policy (destination island)
    const ReplacementPolicy REPLACEMENT_POLICY;
    
    
    /// The number of evolution steps between two migration steps
    const int MIGRATION_PERIOD;
    
    /// The total number of evolution steps after which the genetic algorithm terminates.
    const int NUM_EVOLUTIONS;
    
    // TODO: fix interaction between TSP and Island
    const int GENE_SIZE;
    
    // TODO: maybe make these const
    int numIndividualsSendBuffer;
    
    int numIndividualsReceiveBuffer;
    
    // TODO: allocate these inside constructor (->heap? check this)
    // TODO: deallocate these inside destructor
    int* sendBufferGenes;
    
    double* sendBufferFitness;
    
    int* receiveBufferGenes;
    
    double* receiveBufferFitness;
    
    /// Empties the receive buffers and integrates the data into the current Island data according to the REPLACEMENT_POLICY.
    void emptyReceiveBuffers();
    
    /// Transfers the data stored inside the send buffers according to the MIGRATION_TOPOLOGY in a synchronized
    /// and blocking fashion. The data received is stored inside the receive buffers as the function returns.
    void doSynchronousBlockingCommunication();
    
    /// Uses the SELECTION_POLICY to fill sendBufferFitnesses and sendBufferGenes with data.
    void fillSendBuffers()
    
    /// Computes the size of the send buffer. The size of the send buffer depends on MIGRATION_TOPOLOGY and
    /// NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION. Called once in the constructor.
    int computeSendBufferSize();
    
    /// Computes the size of the receive buffer. The size of the receive buffer differs from NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION
    /// because of the internal use of MPI_Allgather and the case where the immigrants cannot be evenly distributed among
    /// senders. Called once in the constructor.
    int computeReceiveBufferSize(int sendBufferSize); // TODO: sendBufferSize could be accessed via object variable
    
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
                                     int numIndividualsToSample);
    
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
                             int numImmigrants, int** immigrantGenes, double* immigrantFitnesses)
    
};

#endif //DPHPC_PROJECT_ISLAND_HPP
