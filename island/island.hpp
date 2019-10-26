#ifndef DPHPC_PROJECT_ISLAND_HPP
#define DPHPC_PROJECT_ISLAND_HPP

#include "mpi.h" /* requirement for MPI */

#include "travelling_salesman_problem.hpp"


/**
 Facilitates sorting individuals during a migration step
 */
typedef struct Individual {
    
    int idx;
    double fitness;
    
    bool operator<(const Individual& other) const {
        this->fitness < other.fitness;
    }
    
} Individual;


/**
 Helper function to swap the elements of two arrays
 */
template<class T>
void swapArrays(T* arrA, T* arrB, int length) {
    
    for(int idx = 0; idx < length; idx++) {
        
        T tmp = arrA[idx];
        arrA[idx] = arrB[idx];
        arrB[idx] = tmp;
    }
    
}


/**
 Helper function to copy the elements of an array to another one
 */
template<class T>
void copyArray(T* source, T* destination, int length) {
    
    for(int idx = 0; idx < length; idx++) {
        
        destination[idx] = source[idx];
    }
    
}


class Island {
    
public:
    
    /**
     An Island wraps around a TravellingSalesmanProblem
     - Fitness evaluation (C_eval) , cross-over and mutation (C_oper) are done by the underlying TSP
     - The Island adds communication (C_comm)  and performs selection and replacement in this context (C_oper)
     
     \param numTSPNodes number of nodes in the graph (TSP)
     \param populationSize size of the population (TSP)
     \param eliteSize number of individuals that survive for sure (TSP)
     \param mutationRate 1/mutationRate is the probability that an individual gets a mutation (TSP)
     \param migrationPeriod the amount of iterations between two migration steps
     \param migrationAmount the number of individuals each island sends to all others
     \param numPeriods numPeriods * migrationPeriod yields the total number of iterations
    */
    Island(int numTSPNodes, int populationSize, int eliteSize, int mutationRate,
           int migrationPeriod, int migrationAmount, int numPeriods):
    tsp(numTSPNodes, populationSize, eliteSize, mutationRate),
    migrationPeriod(migrationPeriod),
    migrationAmount(migrationAmount),
    numPeriods(numPeriods) {}
    
    /**
     Executes the GA on the current rank. Because of MPI_Allgather it is necessary that all ranks in MPI_COMM_WORLD
     execute the GA simultaneously.
     \return the length of the shortest path found by the algorithm
     */
    double solve();
    
    
private:
    
    /**
     The aggregated TravellingSalesmanProblem
     */
    TravellingSalesmanProblem tsp;
    
    /**
     The amount of iterations of the algorithm between two migration steps
     */
    int migrationPeriod;
    
    /**
     The amount of individuals each island sends to all other islands
     */
    int migrationAmount;
    
    /**
     The number of migration periods. The overall number of iterations is given by numPeriods * migrationPeriod.
     */
    int numPeriods;
    
};

#endif //DPHPC_PROJECT_ISLAND_HPP
